#ifndef PARTICLES_H
#define PARTICLES_H

#include <math.h>

#include "Alloc.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"

struct particles {
    
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    /** maximum number of particles of this species on this domain. used for memory allocation */
    long npmax;
    /** number of particles of this species on this domain */
    long nop;
    
    /** Electron and ions have different number of iterations: ions moves slower than ions */
    int NiterMover;
    /** number of particle of subcycles in the mover */
    int n_sub_cycles;
    
    
    /** number of particles per cell */
    int npcel;
    /** number of particles per cell - X direction */
    int npcelx;
    /** number of particles per cell - Y direction */
    int npcely;
    /** number of particles per cell - Z direction */
    int npcelz;
    
    
    /** charge over mass ratio */
    FPpart qom;
    
    /* drift and thermal velocities for this species */
    FPpart u0, v0, w0;
    FPpart uth, vth, wth;
    
    /** particle arrays: 1D arrays[npmax] */
    FPpart* x; FPpart*  y; FPpart* z; FPpart* u; FPpart* v; FPpart* w;
    /** q must have precision of interpolated quantities: typically double. Not used in mover */
    FPinterp* q;

    
};

/** allocate particle arrays */
void particle_allocate(struct parameters*, struct particles*, int);

/** deallocate */
void particle_deallocate(struct particles*);

#ifdef GPU

/** (de)allocate */
void particle_allocate_device(struct parameters*, struct particles**, int);
void particle_deallocate_device(struct particles*);
/** synchronize */
void particle_synchronize_host(struct particles*, struct particles*);
void particle_synchronize_device(struct particles*, struct particles*);

#endif // GPU

/** particle mover */
int mover_PC(struct particles*, struct EMfield*, struct grid*, struct parameters*);

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles*, struct interpDensSpecies*, struct grid*);

#ifdef GPU

__global__ void mover_PC_kernel(struct particles* part, struct EMField* field, struct grid* grd, struct parameters* param);
__global__ void interpP2G_kernel(struct particles* part, struct interpDensSpecies* ids, struct grid* grd);

#endif // GPU

#endif
