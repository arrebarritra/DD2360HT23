#include "InterpDensSpecies.h"

/** allocated interpolated densities per species */
void interp_dens_species_allocate(struct grid* grd, struct interpDensSpecies* ids, int is)
{
    // set species ID
    ids->species_ID = is;
    
    // allocate 3D arrays
    // rho: 1
    ids->rhon = newArr3<FPinterp>(&ids->rhon_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    ids->rhoc = newArr3<FPinterp>(&ids->rhoc_flat, grd->nxc, grd->nyc, grd->nzc); // center
    // Jx: 2
    ids->Jx   = newArr3<FPinterp>(&ids->Jx_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Jy: 3
    ids->Jy   = newArr3<FPinterp>(&ids->Jy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Jz: 4
    ids->Jz   = newArr3<FPinterp>(&ids->Jz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxx: 5
    ids->pxx  = newArr3<FPinterp>(&ids->pxx_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxy: 6
    ids->pxy  = newArr3<FPinterp>(&ids->pxy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxz: 7
    ids->pxz  = newArr3<FPinterp>(&ids->pxz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pyy: 8
    ids->pyy  = newArr3<FPinterp>(&ids->pyy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pyz: 9
    ids->pyz  = newArr3<FPinterp>(&ids->pyz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pzz: 10
    ids->pzz  = newArr3<FPinterp>(&ids->pzz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    
}

/** deallocate interpolated densities per species */
void interp_dens_species_deallocate(struct grid* grd, struct interpDensSpecies* ids)
{
    
    // deallocate 3D arrays
    delArr3(ids->rhon, grd->nxn, grd->nyn);
    delArr3(ids->rhoc, grd->nxc, grd->nyc);
    // deallocate 3D arrays: J - current
    delArr3(ids->Jx, grd->nxn, grd->nyn);
    delArr3(ids->Jy, grd->nxn, grd->nyn);
    delArr3(ids->Jz, grd->nxn, grd->nyn);
    // deallocate 3D arrays: pressure
    delArr3(ids->pxx, grd->nxn, grd->nyn);
    delArr3(ids->pxy, grd->nxn, grd->nyn);
    delArr3(ids->pxz, grd->nxn, grd->nyn);
    delArr3(ids->pyy, grd->nxn, grd->nyn);
    delArr3(ids->pyz, grd->nxn, grd->nyn);
    delArr3(ids->pzz, grd->nxn, grd->nyn);
    
    
}

#ifdef GPU

/** allocated interpolated densities per species */
void interp_dens_species_allocate_device(struct grid* grd, struct interpDensSpecies* d_ids, int is) {
    interpDensSpecies* ids = new interpDensSpecies;

    // set species ID
    ids->species_ID = is;

    cudaMalloc(&d_ids, sizeof(interpDensSpecies));
    cudaMemcpy(d_ids, ids, sizeof(interpDensSpecies), cudaMemcpyHostToDevice);

    // allocate 3D arrays
    // rho: 1
    newArr3<FPinterp><<<1,1>>>(&d_ids->rhon, &d_ids->rhon_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    newArr3<FPinterp><<<1,1>>>(&d_ids->rhoc, &d_ids->rhoc_flat, grd->nxc, grd->nyc, grd->nzc); // center
    // Jx: 2
    newArr3<FPinterp><<<1,1>>>(&d_ids->Jx, &d_ids->Jx_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Jy: 3
    newArr3<FPinterp><<<1,1>>>(&d_ids->Jy, &d_ids->Jy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Jz: 4
    newArr3<FPinterp><<<1,1>>>(&d_ids->Jz, &d_ids->Jz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxx: 5
    newArr3<FPinterp><<<1,1>>>(&d_ids->pxx, &d_ids->pxx_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxy: 6
    newArr3<FPinterp><<<1,1>>>(&d_ids->pxy, &d_ids->pxy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxz: 7
    newArr3<FPinterp><<<1,1>>>(&d_ids->pxz, &d_ids->pxz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pyy: 8
    newArr3<FPinterp><<<1,1>>>(&d_ids->pyy, &d_ids->pyy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pyz: 9
    newArr3<FPinterp><<<1,1>>>(&d_ids->pyz, &d_ids->pyz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pzz: 10
    newArr3<FPinterp><<<1,1>>>(&d_ids->pzz, &d_ids->pzz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
}

/** deallocate interpolated densities per species */
void interp_dens_species_deallocate_device(struct grid* grd, struct interpDensSpecies* d_ids) {
    
    FPinterp* d_rhon, d_rhoc;
    FPinterp* d_Jx, d_Jy, d_Jz;
    FPinterp* d_pxx, d_pxy, d_pxz, d_pyy, d_pyz, d_pzz;

    cudaMemcpy(&d_rhon, &d_ids->rhon, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_rhoc, &d_ids->rhoc, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jx, &d_ids->Jx, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jy, &d_ids->Jy, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jz, &d_ids->Jz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxx, &d_ids->pxx, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxy, &d_ids->pxy, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxz, &d_ids->pxz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pyy, &d_ids->pyy, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pyz, &d_ids->pyz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pzz, &d_ids->pzz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);

    // deallocate 3D arrays
    delArr3(d_rhon, grd->nxn, grd->nyn);
    delArr3(d_rhoc, grd->nxc, grd->nyc);
    // deallocate 3D arrays: J - current
    delArr3(d_Jx, grd->nxn, grd->nyn);
    delArr3(d_Jy, grd->nxn, grd->nyn);
    delArr3(d_Jz, grd->nxn, grd->nyn);
    // deallocate 3D arrays: pressure
    delArr3(d_pxx, grd->nxn, grd->nyn);
    delArr3(d_pxy, grd->nxn, grd->nyn);
    delArr3(d_pxz, grd->nxn, grd->nyn);
    delArr3(d_pyy, grd->nxn, grd->nyn);
    delArr3(d_pyz, grd->nxn, grd->nyn);
    delArr3(d_pzz, grd->nxn, grd->nyn);

}

void interp_dens_species_synchronize_host(struct interpDensSpecies* h_ids, struct interpDensSpecies* d_ids) {

    cudaMemcpy(d_ids, ids, sizeof(interpDensSpecies), cudaMemcpyHostToDevice);

    FPinterp* d_rhon, d_rhoc;
    FPinterp* d_Jx, d_Jy, d_Jz;
    FPinterp* d_pxx, d_pxy, d_pxz, d_pyy, d_pyz, d_pzz;

    cudaMemcpy(&d_rhon, &d_ids->rhon, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_rhoc, &d_ids->rhoc, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jx, &d_ids->Jx, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jy, &d_ids->Jy, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jz, &d_ids->Jz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxx, &d_ids->pxx, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxy, &d_ids->pxy, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxz, &d_ids->pxz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pyy, &d_ids->pyy, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pyz, &d_ids->pyz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pzz, &d_ids->pzz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);

    cudaMemcpy(ids->rhon, d_rhon, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->rhoc, d_rhoc, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jx, d_Jx, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jy, d_Jy, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jz, d_Jz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxx, d_pxx, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxy, d_pxy, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxz, d_pxz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyy, d_pyy, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyz, d_pyz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pzz, d_pzz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);

}
void interp_dens_species_synchronize_device(struct interpDensSpecies* h_ids, struct interpDensSpecies* d_ids) {

    cudaMemcpy(ids, d_ids, sizeof(interpDensSpecies), cudaMemcpyDeviceToHost);

    FPinterp* d_rhon, d_rhoc;
    FPinterp* d_Jx, d_Jy, d_Jz;
    FPinterp* d_pxx, d_pxy, d_pxz, d_pyy, d_pyz, d_pzz;

    cudaMemcpy(&d_rhon, &d_ids->rhon, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_rhoc, &d_ids->rhoc, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jx, &d_ids->Jx, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jy, &d_ids->Jy, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jz, &d_ids->Jz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxx, &d_ids->pxx, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxy, &d_ids->pxy, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxz, &d_ids->pxz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pyy, &d_ids->pyy, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pyz, &d_ids->pyz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pzz, &d_ids->pzz, sizeof(FPinterp*), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_rhon, ids->rhon, sizeof(FPinterp*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rhoc, ids->rhoc, sizeof(FPinterp*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jx, ids->Jx, sizeof(FPinterp*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jy, ids->Jy, sizeof(FPinterp*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jz, ids->Jz, sizeof(FPinterp*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pxx, ids->pxx, sizeof(FPinterp*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pxy, ids->pxy, sizeof(FPinterp*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pxz, ids->pxz, sizeof(FPinterp*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pyy, ids->pyy, sizeof(FPinterp*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pyz, ids->pyz, sizeof(FPinterp*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pzz, ids->pzz, sizeof(FPinterp*), cudaMemcpyHostToDevice);

}

#endif // GPU

/** deallocate interpolated densities per species */
void interpN2Crho(struct interpDensSpecies* ids, struct grid* grd){
    for (register int i = 1; i < grd->nxc - 1; i++)
        for (register int j = 1; j < grd->nyc - 1; j++)
            for (register int k = 1; k < grd->nzc - 1; k++){
                ids->rhoc[i][j][k] = .125 * (ids->rhon[i][j][k] + ids->rhon[i + 1][j][k] + ids->rhon[i][j + 1][k] + ids->rhon[i][j][k + 1] +
                                       ids->rhon[i + 1][j + 1][k]+ ids->rhon[i + 1][j][k + 1] + ids->rhon[i][j + 1][k + 1] + ids->rhon[i + 1][j + 1][k + 1]);
            }
}
