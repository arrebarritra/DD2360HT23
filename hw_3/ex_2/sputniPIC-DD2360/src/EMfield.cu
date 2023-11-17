#include "EMfield.h"

/** allocate electric and magnetic field */
void field_allocate(struct grid* grd, struct EMfield* field)
{
    // E on nodes
    field->Ex = newArr3<FPfield>(&field->Ex_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Ey = newArr3<FPfield>(&field->Ey_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Ez = newArr3<FPfield>(&field->Ez_flat, grd->nxn, grd->nyn, grd->nzn);
    // B on nodes
    field->Bxn = newArr3<FPfield>(&field->Bxn_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Byn = newArr3<FPfield>(&field->Byn_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Bzn = newArr3<FPfield>(&field->Bzn_flat, grd->nxn, grd->nyn, grd->nzn);
}

/** deallocate electric and magnetic field */
void field_deallocate(struct grid* grd, struct EMfield* field)
{
    // E deallocate 3D arrays
    delArr3(field->Ex, grd->nxn, grd->nyn);
    delArr3(field->Ey, grd->nxn, grd->nyn);
    delArr3(field->Ez, grd->nxn, grd->nyn);

    // B deallocate 3D arrays
    delArr3(field->Bxn, grd->nxn, grd->nyn);
    delArr3(field->Byn, grd->nxn, grd->nyn);
    delArr3(field->Bzn, grd->nxn, grd->nyn);
}

#ifdef GPU

/** allocate electric and magnetic field */
void field_allocate_device(struct grid* grd, struct EMfield* d_field) {
    cudaMalloc(&d_field, sizeof(EMField));

    // E on nodes
    newArr3<FPfield><<<1,1>>>(&d_field->Ex, &d_field->Ex_flat, grd->nxn, grd->nyn, grd->nzn);
    newArr3<FPfield><<<1,1>>>(&d_field->Ey, &d_field->Ey_flat, grd->nxn, grd->nyn, grd->nzn);
    newArr3<FPfield><<<1,1>>>(&d_field->Ez, &d_field->Ez_flat, grd->nxn, grd->nyn, grd->nzn);
    
    // B on nodes
    newArr3<FPfield><<<1,1>>>(&d_field->Bxn, &d_field->Bxn_flat, grd->nxn, grd->nyn, grd->nzn);
    newArr3<FPfield><<<1,1>>>(&d_field->Byn, &d_field->Byn_flat, grd->nxn, grd->nyn, grd->nzn);
    newArr3<FPfield><<<1,1>>>(&d_field->Bzn, &d_field->Bzn_flat, grd->nxn, grd->nyn, grd->nzn);
}

/** deallocate electric and magnetic field */
void field_deallocate_device(struct grid* grd, struct EMfield* d_field) {
    // E deallocate 3D arrays
    FPfield*** d_Ex, *** d_Ey, *** d_Ez;
    cudaMemcpy(&d_Ex, &d_field->Ex, sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Ey, &d_field->Ey, sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Ez, &d_field->Ez, sizeof(FPfield), cudaMemcpyDeviceToHost);

    delArr3_device<<<1,1>>>(d_Ex, grd->nxn, grd->nyn);
    delArr3_device<<<1,1>>>(d_Ey, grd->nxn, grd->nyn);
    delArr3_device<<<1,1>>>(d_Ez, grd->nxn, grd->nyn);

    // B deallocate 3D arrays
    FPfield*** d_Bxn, *** d_Byn, *** d_Bzn;
    cudaMemcpy(&d_Bxn, &d_field->Bxn_flat, sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Byn, &d_field->Byn_flat, sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Bzn, &d_field->Bzn_flat, sizeof(FPfield), cudaMemcpyDeviceToHost);

    delArr3_device<<<1,1>>>(d_field->Bxn_flat, grd->nxn, grd->nyn);
    delArr3_device<<<1,1>>>(d_field->Byn_flat, grd->nxn, grd->nyn);
    delArr3_device<<<1,1>>>(d_field->Bzn_flat, grd->nxn, grd->nyn);
}

/** synchronize */
void particle_synchronize_host(struct grid* grd, struct EMfield* h_field, struct EMfield* d_field) {
    cudaMemcpy(h_field, d_field, sizeof(EMfield), cudaMemcpyDeviceToHost);

    FPfield*** d_Ex_flat, *** d_Ey_flat, *** d_Ez_flat;
    cudaMemcpy(&d_Ex_flat, &d_field->Ex_flat, sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Ey_flat, &d_field->Ey_flat, sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Ez_flat, &d_field->Ez_flat, sizeof(FPfield), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_field->Ex_flat, d_Ex_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_field->Ey_flat, d_Ey_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_field->Ez_flat, d_Ez_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceTHostoHost);

    FPfield*** d_Bxn_flat, d_Byn_flat, d_Bzn_flat;
    cudaMemcpy(&d_Bxn_flat, &(d_field->Bxn_flat), sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Byn_flat, &(d_field->Byn_flat), sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Bzn_flat, &(d_field->Bzn_flat), sizeof(FPfield), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_field->Bxn_flat, d_Bxn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_field->Byn_flat, d_Byn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_field->Bzn_flat, d_Bzn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
}

void particle_synchronize_device(struct grid* grd, struct EMfield* h_field, struct EMfield* d_field) {
    cudaMemcpy(d_field, h_field, sizeof(EMfield), cudaMemcpyDeviceToHost);

    FPfield*** d_Ex_flat, *** d_Ey_flat, *** d_Ez_flat;
    cudaMemcpy(&d_Ex_flat, &(d_field->Ex_flat), sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Ey_flat, &(d_field->Ey_flat), sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Ez_flat, &(d_field->Ez_flat), sizeof(FPfield), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_Ex_flat, h_field->Ex_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ey_flat, h_field->Ey_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ez_flat, h_field->Ez_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);

    FPfield*** d_Bxn_flat, *** d_Byn_flat, *** d_Bzn_flat;
    cudaMemcpy(&d_Bxn_flat, &(d_field->Bxn_flat), sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Byn_flat, &(d_field->Byn_flat), sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Bzn_flat, &(d_field->Bzn_flat), sizeof(FPfield), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_Bxn_flat, h_field->Bxn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Byn_flat, h_field->Byn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bzn_flat, h_field->Bzn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
}

#endif // GPU
