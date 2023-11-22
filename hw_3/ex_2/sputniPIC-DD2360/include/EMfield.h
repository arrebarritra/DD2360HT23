#ifndef EMFIELD_H
#define EMFIELD_H

#include "Alloc.h"
#include "Grid.h"


/** structure with field information */
struct EMfield {
    // field arrays: 4D arrays
    
    /* Electric field defined on nodes: last index is component */
    FPfield*** Ex;
    FPfield* Ex_flat;
    FPfield*** Ey;
    FPfield* Ey_flat;
    FPfield*** Ez;
    FPfield* Ez_flat;
    /* Magnetic field defined on nodes: last index is component */
    FPfield*** Bxn;
    FPfield* Bxn_flat;
    FPfield*** Byn;
    FPfield* Byn_flat;
    FPfield*** Bzn;
    FPfield* Bzn_flat;
    
};

/** allocate electric and magnetic field */
void field_allocate(struct grid*, struct EMfield*);

/** deallocate electric and magnetic field */
void field_deallocate(struct grid*, struct EMfield*);


#ifdef GPU

/** allocate electric and magnetic field */
void field_allocate_device(struct grid*, struct EMfield**);

/** deallocate electric and magnetic field */
void field_deallocate_device(struct grid*, struct EMfield*);

/** synchronize */
void field_synchronize_host(struct grid* grd, struct EMfield*, struct EMfield*);
void field_synchronize_device(struct grid* grd, struct EMfield*, struct EMfield*);

#endif // GPU


#endif
