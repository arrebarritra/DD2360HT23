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
void interp_dens_species_allocate_device(struct grid* grd, struct interpDensSpecies** p_d_ids, int is) {
    interpDensSpecies* ids = new interpDensSpecies;

    // set species ID
    ids->species_ID = is;

    cudaMalloc(p_d_ids, sizeof(interpDensSpecies));
    interpDensSpecies* d_ids = *p_d_ids;
    cudaMemcpy(d_ids, ids, sizeof(interpDensSpecies), cudaMemcpyHostToDevice);

    FPinterp ***d_rhon, ***d_rhoc;
    FPinterp ***d_Jx, ***d_Jy, ***d_Jz;
    FPinterp ***d_pxx, ***d_pxy, ***d_pxz, ***d_pyy, ***d_pyz, ***d_pzz;

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

    cudaMemcpy(&d_ids->rhon, &d_rhon, sizeof(FPinterp***), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_ids->rhoc, &d_rhoc, sizeof(FPinterp***), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_ids->Jx, &d_Jx, sizeof(FPinterp***), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_ids->Jy, &d_Jy, sizeof(FPinterp***), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_ids->Jz, &d_Jz, sizeof(FPinterp***), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_ids->pxx, &d_pxx, sizeof(FPinterp***), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_ids->pxy, &d_pxy, sizeof(FPinterp***), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_ids->pxz, &d_pxz, sizeof(FPinterp***), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_ids->pyy, &d_pyy, sizeof(FPinterp***), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_ids->pyz, &d_pyz, sizeof(FPinterp***), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_ids->pzz, &d_pzz, sizeof(FPinterp***), cudaMemcpyHostToDevice);

    free(ids);

}

/** deallocate interpolated densities per species */
void interp_dens_species_deallocate_device(struct interpDensSpecies* d_ids) {  
    FPinterp*** d_rhon, *** d_rhoc;
    FPinterp*** d_Jx, *** d_Jy, *** d_Jz;
    FPinterp*** d_pxx, *** d_pxy, *** d_pxz, *** d_pyy, *** d_pyz, *** d_pzz;

    cudaMemcpy(&d_rhon, &d_ids->rhon, sizeof(FPinterp***), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_rhoc, &d_ids->rhoc, sizeof(FPinterp***), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jx, &d_ids->Jx, sizeof(FPinterp***), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jy, &d_ids->Jy, sizeof(FPinterp***), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jz, &d_ids->Jz, sizeof(FPinterp***), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxx, &d_ids->pxx, sizeof(FPinterp***), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxy, &d_ids->pxy, sizeof(FPinterp***), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxz, &d_ids->pxz, sizeof(FPinterp***), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pyy, &d_ids->pyy, sizeof(FPinterp***), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pyz, &d_ids->pyz, sizeof(FPinterp***), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pzz, &d_ids->pzz, sizeof(FPinterp***), cudaMemcpyDeviceToHost);

    // deallocate 3D arrays
    delArr3_device(d_rhon);
    delArr3_device(d_rhoc);
    // deallocate 3D arrays: J - current
    delArr3_device(d_Jx);
    delArr3_device(d_Jy);
    delArr3_device(d_Jz);
    // deallocate 3D arrays: pressure
    delArr3_device(d_pxx);
    delArr3_device(d_pxy);
    delArr3_device(d_pxz);
    delArr3_device(d_pyy);
    delArr3_device(d_pyz);
    delArr3_device(d_pzz);
}

void interp_dens_species_synchronize_host(struct interpDensSpecies* h_ids, struct interpDensSpecies* d_ids) {

    cudaMemcpy(d_ids, h_ids, sizeof(interpDensSpecies), cudaMemcpyHostToDevice);

    FPinterp* d_rhon_flat, * d_rhoc_flat;
    FPinterp* d_Jx_flat, * d_Jy_flat, * d_Jz_flat;
    FPinterp* d_pxx_flat, * d_pxy_flat, * d_pxz_flat, * d_pyy_flat, * d_pyz_flat, * d_pzz_flat;

    cudaMemcpy(&d_rhon_flat, &d_ids->rhon_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_rhoc_flat, &d_ids->rhoc_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jx_flat, &d_ids->Jx_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jy_flat, &d_ids->Jy_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jz_flat, &d_ids->Jz_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxx_flat, &d_ids->pxx_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxy_flat, &d_ids->pxy_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxz_flat, &d_ids->pxz_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pyy_flat, &d_ids->pyy_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pyz_flat, &d_ids->pyz_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pzz_flat, &d_ids->pzz_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_ids->rhon_flat, d_rhon_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ids->rhoc_flat, d_rhoc_flat, sizeof(FPinterp) * grd->nxc * grd->nyc * grd->nzc, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ids->Jx_flat, d_Jx_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ids->Jy_flat, d_Jy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ids->Jz_flat, d_Jz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ids->pxx_flat, d_pxx_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ids->pxy_flat, d_pxy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ids->pxz_flat, d_pxz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ids->pyy_flat, d_pyy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ids->pyz_flat, d_pyz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ids->pzz_flat, d_pzz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyDeviceToHost);

}
void interp_dens_species_synchronize_device(struct interpDensSpecies* h_ids, struct interpDensSpecies* d_ids) {

    cudaMemcpy(h_ids, d_ids, sizeof(interpDensSpecies), cudaMemcpyDeviceToHost);

    FPinterp* d_rhon_flat, * d_rhoc_flat,;
    FPinterp* d_Jx_flat, * d_Jy_flat, * d_Jz_flat,;
    FPinterp* d_pxx_flat, * d_pxy_flat, * d_pxz_flat, * d_pyy_flat, * d_pyz_flat, * d_pzz_flat,;

    cudaMemcpy(&d_rhon_flat, &d_ids->rhon_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_rhoc_flat, &d_ids->rhoc_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jx_flat, &d_ids->Jx_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jy_flat, &d_ids->Jy_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_Jz_flat, &d_ids->Jz_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxx_flat, &d_ids->pxx_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxy_flat, &d_ids->pxy_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pxz_flat, &d_ids->pxz_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pyy_flat, &d_ids->pyy_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pyz_flat, &d_ids->pyz_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_pzz_flat, &d_ids->pzz_flat, sizeof(FPinterp*), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_rhon_flat, h_ids->rhon_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rhoc_flat, h_ids->rhoc_flat, sizeof(FPinterp) * grd->nxc * grd->nyc * grd->nzc, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jx_flat, h_ids->Jx_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jy_flat, h_ids->Jy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jz_flat, h_ids->Jz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pxx_flat, h_ids->pxx_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pxy_flat, h_ids->pxy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pxz_flat, h_ids->pxz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pyy_flat, h_ids->pyy_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pyz_flat, h_ids->pyz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pzz_flat, h_ids->pzz_flat, sizeof(FPinterp) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);

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
