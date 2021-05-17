#include <array>
#include <chrono>

#include <itensor/util/h5/group.hpp>
#include <itensor/util/h5/h5.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>

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



#include <TDVP/tdvp.h>
#include <TDVP/basisextension.h>


using namespace std::chrono;
using namespace itensor;


// TODO: 
//        track norm in TDVP 
//        mpi parallel
//        make tdvp work with heff
//        check that cutoff and maxm is used by sweeps tdvp


using Dvec = std::vector<double>;
using Dmat = std::vector<Dvec>;
using opPair = std::pair<std::string, std::string>;

// Simple container for a MPO and a name for it
struct MPO_Observable {
  // simple struct to combine an operator and its name
  MPO mpo;
  std::string name;
};

// Class used if no extra measurement should be performed
class NoExtraMeasurement {
  public:
  static void measure(MPS& psi){} // do nothing
};

// Class used if no Green function should be computed
class NoGF {
  public: 
  static void apply(MPS& metts){};     // do nothing
  static void applyDag(MPS& metts) {}; // do nothing
};





/** General purpose ITensor METTS implementation.
 * Optionally also allows to compute imaginary time correlation functions <A(τ)B>
 *
 * The time evolutions are performed using the Time Dependent Variational Principle (TDVP)
 * (https://github.com/ITensor/TDVP) using global subspace expansion (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.094315)
 *
 * The user only has to provide four things that are discussed in more detail below:
 * - The Model including the Hamiltonian, the SiteSet and a starting product state.
 * - A projector: given a metts, how to project to a new product state.
 * - What to measure.
 * - How do the operators A and B act (in case a GF should be computed).
 *
 *
 *
 *
 * The model is defined using the initModel(...) function
 * setting the Hamiltonian, the SiteSet as well as the first Metts that 
 * should be used.
 *
 *
 *
 *
 * The projection from a metts to a new product state is model and method 
 * dependent as you can for example mix purification and metts (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.101.195119).
 * Therefore, the user has to define what it means to project.
 * This happens through a user-defined class that must have a function:
 *        MPS project(MPS& metts).
 * This function takes the metts state as argument and returns the new product state.
 *
 *
 *
 *
 * Measurements of static observables can be performed in two ways:
 * 1. Default Measurements:
 *    Include single-site measurements, nearest neighbor measurements and measurements based on an MPO.
 *    Those are defined via the respective setObservables functions (see sample/example.cc):
 *    1.1 setSingleSiteObservables(...):
 *        Single-site measurements are defined via a list of strings that contain 
 *        the operator name in the SiteSet.
 *        
 *    1.2 setNearesNeighborObservables(...)
 *        Neares-neighbor measurements are defined via a list of pairs of strings 
 *        that contain the operator name in the SiteSet.
 *
 *    1.3 setMPOObservables(...)
 *        Called by providing a list of 'MPO_Observables' object. This is a simple 
 *        struct containing a mpo and a string that defines its name.
 *    
 *    Default Measurements are automatically saved to an HDF5 archive with a 
 *    user defined name.    
 *
 * 2. Extra Measurements:
 *    User gets access to the metts via an extraMeasurement class. Through that, 
 *    any measurement desired can be performed. The extra measurement class must 
 *    have a function:
 *        measure(MPS metts); // no reference here
 * 
 *    Extra Measurements are currently not saved to the HDF5 
 *    archive and need to be saved by the user after completion. The reason is 
 *    that the user should not be forced to know how to use HDF5. Instead, 
 *    measurements should be stored inside the Extra Measurement object which can be
 *    accessed via the MettsSolver::extraMeasure member. 
 * 
 *
 *
 * The action of the operators A and B is defined via a two instances of a 
 * class that has two functions:
 *    void apply(MPS& metts)
 *    void applyDag(MPS& metts)
 * which define the action of the operator itself as well as the action of the Hermitian
 * conjugate. The latter is required to obtain a measurement of the Green function
 * that does not diverge in τ.
 *
 *
 * Depending on wheter a Green function should be computed at all and if an extra 
 * Measurement is necessary, there are 4 constructors available. All arguments
 * other than args are templates:
 * 
 * Compute Green function <A(τ)B>:
 *  Include extra measurement:
 *    MettsSolver(const Args& args, GFOp A, GFOp B, Projector p, ExtraMeasurement M);
 *  No extra measurement:
 *    MettsSolver(const Args& args, GFOp A, GFOp B, Projector p);
 *
 * Do NOT compute Green function:
 *  Include extra measurement:
 *    MettsSolver(const Args& args, Projector p, ExtraMeasurement M);
 *  No extra measurement:
 *    MettsSolver(const Args& args, Projector p);
 *
 * So in the simplest case, one only needs to provide a Projector p to perform the
 * METTS calculation.
 *
 * The parameters used by the computation are stored in the ITensor Args object.
 * Currently used in args are:
 * "beta":          double        
 * Inverse Temperature.
 *
 * "NMetts":        int (default 2000)
 * Number of Metts computed.
 *
 * "NTherm":        int (default 20)
 * Number of thermalization steps.
 *
 * "dtau":          double (default 0.5)
 * Time step size of TDVP imaginary time evolution.
 *
 * "NExpand":       int (default 2)
 * Number of steps global subspace expansion is performed.
 *
 * "outfileName":    string 
 * Default measurement results and Green function is stored in a HDF5 archive with name 'outfileName.h5'
 *
 * "MaxDim":          int 
 * Maximum bond dimension.
 *
 * "Cutoff":          double 
 * Truncated weight.
*/


template<typename GFOp, typename Projector, typename ExtraMeasurement>
class MettsSolver {


  public:
  // Metts solver computing Green function <A(τ) Β> with projector p and extra measurement M
  MettsSolver(const Args& args, GFOp A, GFOp B, Projector p, ExtraMeasurement M): 
    args(args),
    beta(args.getReal("beta")),
    phitau(0),
    NMetts( args.getInt("NMetts", 2000) ),
    NTherm(args.getInt("NTherm", 20) ),
    projector(p),
    dtau(args.getReal("dtau",0.5)),
    NExpand(args.getReal("NExpand",2)),
    Op_A(A),
    Op_B(B),
    outfileName(args.getString("outFileName")),
    Gfs(0),
    extraMeasure(M)
  {
    init();
  }

  // Metts solver computing Green function <A(τ) Β> with projector p without any extra measurement
  MettsSolver(const Args& args, GFOp A, GFOp B, Projector p): 
    args(args),
    beta(args.getReal("beta")),
    phitau(0),
    NMetts( args.getInt("NMetts", 2000) ),
    NTherm(args.getInt("NTherm", 20) ),
    projector(p),
    dtau(args.getReal("dtau",0.5)),
    NExpand(args.getReal("NExpand",2)),
    Op_A(A),
    Op_B(B),
    outfileName(args.getString("outFileName")),
    Gfs(0),
    extraMeasure(NoExtraMeasurement())
  {
    init();
  }

  // Metts with projector p with extra measurement M but no Green function
  MettsSolver(const Args& args, Projector p, ExtraMeasurement M): 
    args(args),
    beta(args.getReal("beta")),
    phitau(0),
    NMetts( args.getInt("NMetts", 2000) ),
    NTherm(args.getInt("NTherm", 20) ),
    projector(p),
    dtau(args.getReal("dtau",0.5)),
    NExpand(args.getReal("NExpand",2)),
    doGF(false),
    Op_A(NoGF()),
    Op_B(NoGF()),
    outfileName(args.getString("outFileName")),
    Gfs(0),
    extraMeasure(M)
  {
    init();
  }


  // Metts with projector p without any extra measurement and no Green function
  MettsSolver(const Args& args, Projector p): 
    args(args),
    beta(args.getReal("beta")),
    phitau(0),
    NMetts( args.getInt("NMetts", 2000) ),
    NTherm(args.getInt("NTherm", 20) ),
    projector(p),
    dtau(args.getReal("dtau",0.5)),
    NExpand(args.getReal("NExpand",2)),
    doGF(false),
    Op_A(NoGF()),
    Op_B(NoGF()),
    outfileName(args.getString("outFileName")),
    Gfs(0),
    extraMeasure(NoExtraMeasurement())
  {
    init();
  }




  void initModel(SiteSet& siteset, MPO& mpo, MPS& firstProductState){
    sites = siteset; 
    metts = firstProductState;
    H = mpo;

    inited = true;
  }

  void setSingleSiteObservables(const std::vector<std::string>& ops){
    std::cout << "\n\nSetting single site observables:\n";

    // check that operators exist in SiteSet, ITensor terminates with an error message if it does not
    for(const auto& op : ops){
      sites.op(op, length(sites));

      std::cout << "   Adding Operator " + op + "\n";
      SS_Results[op] = Dmat();
    }

    SS_Operators = ops;
  }

  void setNearesNeighborObservables(const std::vector<opPair> opPairs){
    std::cout << "\nSetting single site observables:\n";
    for(const auto& [op1, op2] : opPairs){
      sites.op(op1, length(sites));
      sites.op(op2, length(sites));

      std::cout << "   Adding Operators " + op1 + " " + op2 + "\n";
      NN_Results[op1 + op2] = Dmat();
    }

    NN_Operators = opPairs;
  }

  void setMPOObservables(const std::vector<MPO_Observable> ops){
    std::cout << "\nSetting MPO observables:\n";

    for(const auto& op : ops){
      std::cout << "   Add observable " + op.name +"\n";
      MPO_Results[op.name] = Dvec();
    }

    MPO_Observables = ops;
  }

  void run(){
    if(!inited) Error("Please initialize the model and starting state using the function initModel() before calling run().");

    HeffMetts = LocalMPO(H);
    //thermalize();

    HeffGr = LocalMPO(H);
    HeffLe = LocalMPO(H);

    for(auto step : range1(NMetts)){
      auto start = high_resolution_clock::now();

      generateMetts();
      if(doGF) calcGF();
      defaultMeasure();
      extraMeasure.measure(metts);

      metts = projector.project(metts);

      auto duration = duration_cast<seconds>(high_resolution_clock::now() - start);
      executionTimes.push_back(duration.count());

      if(step%10==0)
        std::cout<< "   Steps done: " << step  <<std::endl; 
    }

    writeToH5();
    
    
  }




  private:
  void init(){
    // time steps
    stepsTotal = int( std::round( beta/(2.*dtau) ) );
    phitau.resize(stepsTotal+1);

    if( std::abs( beta/2. - stepsTotal*dtau  ) > 1E-15 ){
      Print(beta);
      Print(dtau);
      throw std::runtime_error("beta/2 cannot be divided by dtau.");
    }


    //output
    //outfileName += "_rank" + std::to_string(world.rank()) + ".h5";
    outfileName += ".h5";
    auto arch = h5::file(outfileName, 'w');
  }


  // perform imaginary time evolution to beta/2 to compute the METTS
  void generateMetts(){
    // reset Eff-H
    HeffMetts = LocalMPO(H);

    double currentNorm = norm(metts);
    double totalT = 0;
    if(doGF) phitau[0] = {metts, currentNorm};
    
    auto sweeps = Sweeps(1);
    sweeps.niter() = 50;

    for(auto i : range1(stepsTotal)){

      if(i <= NExpand)
            {
            // Global subspace expansion
            std::vector<Real> epsilonK = {1E-12, 1E-12, 1E-12};
            addBasis(metts, H, epsilonK,{ "Cutoff",1E-8,
                                          "Method","DensityMatrix",
                                          "KrylovOrd",3,
                                          "DoNormalize",true,
                                          "Silent",true,
                                          "Quiet",true});
            }

      tdvp(metts, H, -dtau, sweeps, {args, "DoNormalize", false, 
                                           "Silent",      true,
                                           "NumCenter",   2});
      totalT += dtau;
      currentNorm = metts.normalize();
      if(currentNorm < 1E-10 or currentNorm > 1E10){
        std::cout<< "Warning, norm during imaginary time evolution very large or very small. " 
                 << "Maybe an energy shift H -> H-E0 can make the compuation more stable. Please check your results.\n";
      }

      //std::cout<< "TDVP: "<<currentNorm<<std::endl;
      if(doGF) phitau.at(i)  = {metts, currentNorm};
    }

    if(abs(totalT - beta/2.)>1E-12){
      std::cout << totalT << " " << beta/2. << " " << totalT - beta/2.;
      Error("Missmatch in time steps");
    }
  
  }

  // thermalization
  void thermalize(){
    std::cout << "Thermalizing"<<std::endl;
    for(auto step : range1(NTherm)){
      HeffMetts = LocalMPO(H);
      generateMetts();  
      
      metts = projector.project(metts);
    }
    std::cout << "Thermalizing Done"<<std::endl;
  }

  // default measurements
  void defaultMeasure(){
    // Single-Site Measurements
    for(const auto& op : SS_Operators){
      Dvec result;
      for(auto site : range1(length(sites))){
        metts.position(site);

        auto val = metts.A(site) * sites.op(op,site) * dag(prime(metts.A(site),"Site"));
        result.push_back( std::real(val.cplx()) );
      }

      SS_Results[op].emplace_back(std::move(result));
    }


    // Neares-Neighbor measurements
    for(const auto& [op1, op2] : NN_Operators){
      Dvec result;
      for(auto site : range1(length(sites)-1)){
        metts.position(site);
        auto link = commonIndex(metts.A(site), metts.A(site+1));

        auto AA = metts.A(site) * metts.A(site+1);
        AA *= sites.op(op1, site);
        AA *= sites.op(op2, site+1);

        AA *= dag(prime(metts.A(site),   "Site"));
        AA *= dag(prime(metts.A(site+1), "Site"));

        result.push_back( std::real(AA.cplx()) );
      }

      NN_Results[op1+op2].emplace_back( std::move(result) );
    }

    // MPO measurements
    for(const auto& op : MPO_Observables){
      MPO_Results[op.name].push_back( inner(metts,op.mpo,metts) );
    }



  }

  // Application of operators, the flag reused defines if it has to act 
  // on the states that can be reused or not.
  void applyGFOp(MPS& bra, MPS& ket, bool reused){
    // Greens functions are computed as:
    //
    //                             reuse and apply Adag               apply B and tevo
    //                                    ∧                                 ∧
    //                                   /|\                               /|\
    //                                    |                                 |
    //                                    |                                 |
    // G+(t) = <A(t) B>      = ( A^dag exp(t H) | m > )^dag       ( exp(-t H ) B | m > ) 
    //
    // G-(t) = <B A(t-beta)> = ( exp(-(b-t) H) B^dag | m > )^dag  ( A exp((t-b) H) | m >) 
    //                                    |                                 |
    //                                    |                                 |
    //                                   \|/                               \|/ 
    //                                    ∨                                 ∨
    //                             apply Bdag and tevo              reuse and apply A

    if(reused){
      // bra gets Adag (bra of G+)
      // ket gets A    (ket of G-)
      Op_A.applyDag(bra);
      Op_A.apply(ket);
    }
    else{
      // bra gets Bdag (bra of G-)
      // ket gets B    (ket of G+)
      Op_B.applyDag(bra);
      Op_B.apply(ket);
    }

  }
  
  // computes the Green function
  void calcGF(){

    // time evolve the ket state for "+" and the bra state for the "-" part of 
    MPS braLe = metts, ketGr = metts;
    applyGFOp(braLe, ketGr, false);

    // also reset Effective Hamiltonians
    HeffLe = LocalMPO(H);
    HeffGr = LocalMPO(H);

    Dvec normsGr(0);
    Dvec overlapGr(0);

    int NTimeStepsGF = int( std::round(beta/dtau) );
    int StepsGr = static_cast<int>( NTimeStepsGF/2 ); 

    Gfs.resize( Gfs.size() + 1 );
    auto& gf = Gfs.back();
    gf.resize(NTimeStepsGF+1,0.);

    if( std::abs( norm(ketGr) ) < 1E-15 || std::abs( norm(braLe)) < 1E-15 ){
      Error("0-norm of ketGr or braLe!!!");
      return ;
    }

    // index at which we take the thermal states stored in phitau
    int phitau_index = phitau.size() - 1;
    // index of the Green's function entries
    int GfIndx = 0;

    double normGr = ketGr.normalize();
    double normLe = braLe.normalize();

    auto measureGf = [&](){
      auto braGr = phitau.at(phitau_index).state;
      auto ketLe = phitau.at(phitau_index).state;
      applyGFOp(braGr, ketLe, true);

      auto valGr = inner(braGr, ketGr)*normGr;
      auto valLe = inner(braLe, ketLe)*normLe;
      
      if( GfIndx == StepsGr){
        // average at beta /2
        gf.at(GfIndx) = 0.5*(valGr+valLe);
      }
      else {
        gf.rbegin()[GfIndx] =  valLe;
        gf.at(GfIndx) =  valGr;
      }

      return; 
    }; 

    measureGf();
    

    auto sweeps = Sweeps(1);
    sweeps.niter() = 50;


    for(auto step : range(StepsGr)){
      tdvp(ketGr, H, -dtau, sweeps, {args, "DoNormalize", false, 
                                      "Silent",      true,
                                      "NumCenter",   2});

      normGr *= ketGr.normalize()/phitau.at(phitau_index).normFactor;

      tdvp(braLe, H, -dtau, sweeps, {args, 
                                     "DoNormalize", false, 
                                     "Silent",      true,
                                     "NumCenter",   2});

      normLe *= braLe.normalize()/phitau.at(phitau_index).normFactor;

      phitau_index--;
      GfIndx++;

      measureGf();
    }



  }

  // writes default measurements and Green function to H5
  void writeToH5(){
    auto arch = h5::file(outfileName, 'a');
    h5_write(arch, "Gf", Gfs);

    h5::group basegroup{arch};

    // write single-site observables
    auto ss_group = basegroup.create_group("SS_Observables");
    for(const auto& op : SS_Operators)
      h5_write(ss_group, op, SS_Results[op]);


    // write TS-site nearest neighbor observables
    auto nn_group = basegroup.create_group("NN_Observables");
    for(const auto& [op1, op2] : NN_Operators)
      h5_write(nn_group, op1 + op2, NN_Results[op1+op2]);
    

    // write MPO observables
    auto mpo_group = basegroup.create_group("MPO_Observables");
    for(const auto& op : MPO_Observables)
      h5_write(mpo_group, op.name, MPO_Results[op.name]);
    

    h5_write(arch, "Times", executionTimes);
  }
  
  


  private:
  /// Mpi Communicator
  //mpi::communicator world;

  Args args;
  double beta;

  struct ImagTimeState {
    // A time evolved state and a factor that was used to normalize it.
    MPS state;
    double normFactor;
  };
  // store states created during computation of the metts
  std::vector<ImagTimeState> phitau; 


  MPS metts; 
  MPO H; 

  SiteSet sites; 

  // QMC stuff
  int NMetts;
  int NTherm;

  Projector projector;

  bool inited = false;

  // time to compute a Metts and the GF
  std::vector<int> executionTimes = {};
  

  // Local operators for TDVP
  LocalMPO HeffMetts = {}; 
  LocalMPO HeffGr = {}; 
  LocalMPO HeffLe = {}; 



  // Time evolution parameters
  int stepsTotal; 
  double dtau; 
  int NExpand; 


  // flag deciding if GF is actually computed
  bool doGF = true;

  // operators defining the Green function, need to have the functions apply and applyDag implemented 
  GFOp Op_A, Op_B;


  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // --------------------------- Measurment results ----------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  std::string outfileName; 
  Dmat Gfs; 

  // operators to measure on each site of the state, operator must be defined in SiteSet
  std::vector<std::string> SS_Operators = {};
  std::map<std::string, Dmat> SS_Results = {};

  // Two-site operators to measure on sites i and i+1
  std::vector< opPair > NN_Operators = {};
  std::map<std::string, Dmat> NN_Results = {};

  // MPO-observables
  std::vector<MPO_Observable> MPO_Observables = {};
  std::map<std::string, Dvec> MPO_Results = {};

  public:
  // make it public so the user can access it after computations
  ExtraMeasurement extraMeasure;
};


// template deduction
template<typename GFOp, typename Projector>
MettsSolver(const Args& args, GFOp A, GFOp B, Projector p) -> MettsSolver<GFOp, Projector, NoExtraMeasurement>;

template<typename Projector, typename ExtraMeasurement>
MettsSolver(const Args& args, Projector p, ExtraMeasurement M) -> MettsSolver<NoGF, Projector, ExtraMeasurement>;

template<typename Projector>
MettsSolver(const Args& args, Projector p) -> MettsSolver<NoGF, Projector, NoExtraMeasurement>;


