/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);

#ifdef GPU

    parameters* d_param;
    cudaMalloc(&d_param, sizeof(parameters));
    cudaMemcpy(d_param, &param, sizeof(parameters), cudaMemcpyHostToDevice);

#endif // GPU
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
#ifdef GPU
    grid* d_grd;
    setGrid_device(&param, &d_grd);
#endif // GPU

    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd,&field);

#ifdef GPU
    EMfield* d_field;
    field_allocate_device(&grd, &d_field);
#endif // GPU

    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
#ifdef GPU
    interpDensSpecies **d_ids = new interpDensSpecies*[param.ns];
#endif // GPU


    for (int is = 0; is < param.ns; is++) {
        interp_dens_species_allocate(&grd, &ids[is], is);

#ifdef GPU
        interp_dens_species_allocate_device(&grd, &d_ids[is], is);
#endif // GPU

    }
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];

#ifdef GPU    
    particles **d_part = new particles*[param.ns];
#endif // GPU

    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);

#ifdef GPU
        particle_allocate_device(&param, &d_part[is], is);
#endif // GPU
    }
    
    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);
#ifdef GPU
    for (int is = 0; is < param.ns; is++) {
        particle_synchronize_device(&part[is], d_part[is]);
        interp_dens_species_synchronize_device(&ids[is], d_ids[is]);
    }
    field_synchronize_device(&grd, &field, d_field);
#endif // GPU

    
    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);
        
        
        
        // implicit mover
        iMover = cpuSecond(); // start timer for mover
        for (int is = 0; is < param.ns; is++) {
#ifndef GPU
            mover_PC(&part[is], &field, &grd, &param);
#else
            mover_PC(d_part[is], d_field, d_grd, d_param);
#endif // GPU
        }

        eMover += (cpuSecond() - iMover); // stop timer for mover
        
        

        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is = 0; is < param.ns; is++) {
#ifndef GPU
            interpP2G(&part[is], &ids[is], &grd);
#else
            interp_dens_species_synchronize_device(&ids[is], d_ids[is]);
            interpP2G(d_part[is], d_ids[is], d_grd);
            interp_dens_species_synchronize_host(&ids[is], d_ids[is]);            
#endif // GPU
        }

        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        
        
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        
    
    }  // end of one PIC cycle
    
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }
    
    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}


