#include <array>
#include <chrono>
#include <random>



#include <algorithm>
#include <cmath>
#include <fstream>

#include <iomanip>
#include <itensor/all.h>
#include <itensor/itensor.h>
#include <itensor/mps/localmpo.h>
#include <itensor/mps/mps.h>
#include <itensor/mps/sites/spinhalf.h>
#include <itensor/mps/siteset.h>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>
#include <complex>

#include "../MettsSolver.hpp"



// Defines any extra measurements performed on the metts state. 
// An extra measurement is optional and can just be left out in the construction 
// of the MettsSolver.
// Must have a function  
//    void measure(MPS& metts, const SiteSet& sites)
// which performs all the extra measurements one wants to perform
// and stores the result.

class MyExtraMeasurement{
  public:

  MyExtraMeasurement(SiteSet& sites):
  sites(sites),
  SzSz()
  {}

  void measure(MPS metts){

    // compute Sz_1 Sz_i correlator for i = 2 ... N
    metts.position(1);
    auto E = metts.A(1) * sites.op("Sz", 1);
    E *= dag(prime(metts.A(1)));

    SzSz.push_back({});

    for(auto site : range1(2, length(sites))){
      // multiply first matrix onto E
      E *= metts.A(site);

      // measure
      auto measure = E * sites.op("Sz", site);
      auto link = commonIndex(metts.A(site), metts.A(site-1));
      measure *= dag( prime(metts.A(site),link, sites.si(site)) );

      SzSz.back().push_back( std::real(measure.cplx()) );

      // translate E by one site
      E *= sites.op("Id", site); // not the most efficient way to do it, but the code is nice
      E *= dag(prime(metts.A(site)));
    }
  }

  SiteSet sites;
  // container storing the results
  Dmat SzSz;
};


// Defines what it means to apply the Greens function operator.
// Computing Green functions is optional and can just be left out in the construction 
// of the MettsSolver.
// 
// If specified, an operator must have two functions:
//     void apply(MPS& state) 
//     void applyDag(MPS& state) 
// defining its action and the action of the hermitian conjugate.
class SingleSiteGFOp{
  
  public:
  SingleSiteGFOp(ITensor op, int s): op(op), site(s)
  {
    // compute hermitian conjugate
    opdag = op;
    opdag.swapPrime(0,1);
    opdag = dag(opdag);
  };

  SingleSiteGFOp() = default;



  void apply(MPS& state){
    if(!op) Error("Trying to apply default constructed operator");

    state.position(site);
    auto& A = state.Aref(site);
    A *= op;
    A.noPrime();
  }

  void applyDag(MPS& state){    
    if(!opdag) Error("Trying to apply h.c. of default constructed operator");

    state.position(site);
    auto& A = state.Aref(site);
    A *= opdag;
    A.noPrime();
  }

  ITensor op = {};
  ITensor opdag = {};
  int site = {};
};


// Defines what it means to project metts to a new product state.
// must have one function:
//     MPS project(MPS& state) 
// which returns the new product state
class SpinHalfProjector{
  // In this example we only perform a Sz-projection, meaning that we 
  // sample from the cannonical ensemble with fixed magnetization
  public:

  SpinHalfProjector(SiteSet sites): 
  sites(sites), 
  init(sites),
  Uniform(0.,1.) 
  { 
    std::random_device rd; 
    RNG = std::mt19937(rd());
  };


  MPS project(MPS& metts){
    init = InitState(sites); // reset init state

    for(auto i : range1(length(sites))){
      metts.position(i);
      
      auto A = metts.A(i);

      // decide onto which state to project on site i
      auto siteProjector = DecideOutcome(A, i);

      // multiply site Projector
      A *= siteProjector;
      A.noPrime();
      metts.Aref(i) = A;
    }

    return MPS(init);
  }


  private: 

  ITensor DecideOutcome(const ITensor& A, int i){
    std::string outcome = "";
    double prob = 0;
    std::string Op = "Sz";

    // measure Operator
    auto res = std::real((A* sites.op(Op, i) * dag(prime(A,"Site"))).cplx());
    res += 0.5;

    auto randomNumber = Uniform(RNG);
    if(res > randomNumber){
      prob = res;
      outcome = "Up";
    }
    else {
      prob = 1-res;
      outcome = "Dn"; 
    }
    
    init.set(i, outcome);

    // generate projector
    Index s = dag(sites.si(i));
    Index sp = sites.siP(i);

    auto siteProjector = ITensor(s, sp);
    siteProjector.set( dag(sites(i, outcome)), prime(sites(i, outcome)), 1.  );
    siteProjector *=  1./std::sqrt(prob);

    return siteProjector;
  }

  SiteSet sites;
  InitState init;
  
  std::mt19937 RNG;
  std::uniform_real_distribution<double> Uniform;

};




int main (int argc, char* argv[]) {

  if(argc != 2) { 
   printfln("Usage: %s inputfile",argv[0]); 
   return 0; 
  }

  // read in parameters
  std::string inputGroup="";
  if(argc != 3) inputGroup = "input";
  else {
      inputGroup = argv[2];
  }
  auto input = InputGroup(argv[1],inputGroup);

  auto U = input.getReal("U"); // interaction
  auto N  = input.getReal("N");

  auto beta = input.getReal("beta"); //beta
  auto dtau = input.getReal("dtau"); // TDVP time step

  auto NMetts = input.getInt("NMetts");
  auto NTherm = input.getInt("NTherm");

  auto tw  = input.getReal("Cutoff",  1E-10);
  auto maxm  = input.getReal("MaxDim",  100);
  
  auto outFileName = input.getString("outfile");

  Args args { "Cutoff", tw, 
              "beta", beta,
              "MaxDim", maxm,
              "dtau", dtau,
              "NMetts",NMetts,
              "NTherm",NTherm,
              "outFileName", outFileName
            };
  



  // ----------------------  Model ----------------------

  // SiteSet
  SpinHalf sites(N);
  
  InitState init(sites);
  for(auto i : range1(length(sites))){
    //init.set(i, "Up");
    i%2 == 1 ? init.set(i, "Up") : init.set(i, "Dn");
  }
  MPS firstMetts(init);

  // H
  auto ampo = AutoMPO(sites);
  for(int j = 1; j < N; ++j){
      ampo += 0.5,"S+",j,"S-",j+1;
      ampo += 0.5,"S-",j,"S+",j+1;
      ampo +=     "Sz",j,"Sz",j+1;
  }
  auto H = toMPO(ampo);



  MyExtraMeasurement extraMeasure(sites);

  //auto var = MPI_Init(NULL, NULL);
  SingleSiteGFOp A(sites.op("Sz", 1), 1), B(sites.op("Sz", 1), 1);


  // Create a MettsSolver
  MettsSolver mettsSolver(args, A, B, SpinHalfProjector(sites), extraMeasure);
  // Alternatively:
  // MettsSolver mettsSolver(args, A, B, SpinHalfProjector(sites)); // no extra measurements
  // MettsSolver mettsSolver(args, SpinHalfProjector(sites), extraMeasure); // no Green function
  // MettsSolver mettsSolver(args, SpinHalfProjector(sites)); // no Green function and no extra measurements

  // set model
  mettsSolver.initModel(sites, H, firstMetts);
  // set single-site measurements
  mettsSolver.setSingleSiteObservables( {"Sz", "Sp"} );
  // set nearest neighbor measurements (measure Sz-Sz correlation)
  mettsSolver.setNearesNeighborObservables( {{"Sz", "Sz"}} );
  // set observales defined via an MPO
  mettsSolver.setMPOObservables( {{H, "energy"}} );

  // Run it
  mettsSolver.run();

  return 0;

}
