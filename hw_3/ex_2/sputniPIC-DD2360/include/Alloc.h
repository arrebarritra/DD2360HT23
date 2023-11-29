#ifndef Alloc_H
#define Alloc_H
#include <cstdio>
#include <cuda_runtime.h>

__host__ __device__
inline long get_idx(long v, long w, long x, long y, long z, long stride_w, long stride_x, long stride_y, long stride_z)
{
	return stride_x * stride_y * stride_z * w + stride_y * stride_z * x + stride_z * y + z;
}

__host__ __device__
inline long get_idx(long w, long x, long y, long z, long stride_x, long stride_y, long stride_z)
{
	return stride_x * stride_y * stride_z * w + stride_y * stride_z * x + stride_z * y + z;
}

__host__ __device__
inline long get_idx(long x, long y, long z, long stride_y, long stride_z)
{
	return stride_y * stride_z * x + stride_z * y + z;
}

__host__ __device__
inline long get_idx(long x, long y, long s1)
{
	return x + (y * s1);
}

template < class type >
inline type* newArr1(size_t sz1)
{
	type* arr = new type[sz1];
	return arr;
}

template < class type >
inline type* newArr1_device(size_t sz1)
{
	type* arr;
	cudaMalloc(&arr, sizeof(type) * sz1);
	return arr;
}

template < class type >
inline type** newArr2(size_t sz1, size_t sz2)
{
	type** arr = new type * [sz1]; // new type *[sz1];
	type* ptr = newArr1<type>(sz1 * sz2);
	for (size_t i = 0; i < sz1; i++)
	{
		arr[i] = ptr;
		ptr += sz2;
	}
	return arr;
}

template < class type >
inline type** newArr2_device(size_t sz1, size_t sz2)
{
	type** arr;
	cudaMalloc(&arr, sizeof(type*) * sz1); // new type *[sz1];
	type* ptr = newArr1_device<type>(sz1 * sz2);
	type* ptrarr[sz1];
	for (size_t i = 0; i < sz1; i++) {
		ptrarr[i] = ptr;
		ptr += sz2;
	}
	cudaMemcpy(arr, ptrarr, sizeof(type*) * sz1, cudaMemcpyHostToDevice);
	return arr;
}

template < class type >
inline type*** newArr3(size_t sz1, size_t sz2, size_t sz3)
{
	type*** arr = new type * *[sz1]; // new type **[sz1];
	type** ptr = newArr2<type>(sz1 * sz2, sz3);
	for (size_t i = 0; i < sz1; i++)
	{
		arr[i] = ptr;
		ptr += sz2;
	}
	return arr;
}

template < class type >
inline type*** newArr3_device(size_t sz1, size_t sz2, size_t sz3)
{
	type*** arr;
	cudaMalloc(&arr, sizeof(type**) * sz1); // new type **[sz1];
	type** ptr = newArr2_device<type>(sz1 * sz2, sz3);
	type** ptrarr[sz1];
	for (size_t i = 0; i < sz1; i++)
	{
		ptrarr[i] = ptr;
		ptr += sz2;
	}
	cudaMemcpy(arr, ptrarr, sizeof(type**) * sz1, cudaMemcpyHostToDevice);
	return arr;
}

template <class type>
inline type**** newArr4(size_t sz1, size_t sz2, size_t sz3, size_t sz4)
{
	type**** arr = new type * **[sz1]; //(new type ***[sz1]);
	type*** ptr = newArr3<type>(sz1 * sz2, sz3, sz4);
	for (size_t i = 0; i < sz1; i++) {
		arr[i] = ptr;
		ptr += sz2;
	}
	return arr;
}

template <class type>
inline type**** newArr4_device(size_t sz1, size_t sz2, size_t sz3, size_t sz4)
{
	type**** arr;
	cudaMalloc(&arr, sizeof(type***) * sz1); //(new type ***[sz1]);
	type*** ptr = newArr3_device<type>(sz1 * sz2, sz3, sz4);
	type*** ptrarr[sz1];
	for (size_t i = 0; i < sz1; i++) {
		ptrarr[i] = ptr;
		ptr += sz2;
	}
	cudaMemcpy(arr, ptrarr, sizeof(type***) * sz1, cudaMemcpyHostToDevice);
	return arr;
}

// build chained pointer hierarchy for pre-existing bottom level
//

/* Build chained pointer hierachy for pre-existing bottom level                        *
 * Provide a pointer to a contig. 1D memory region which was already allocated in "in" *
 * The function returns a pointer chain to which allows subscript access (x[i][j])     */
template <class type>
inline type***** newArr5(type** in, size_t sz1, size_t sz2, size_t sz3, size_t sz4, size_t sz5)
{
	*in = newArr1<type>(sz1 * sz2 * sz3 * sz4 * sz5);

	type***** arr = newArr4<type*>(sz1, sz2, sz3, sz4);
	type** arr2 = ***arr;
	type* ptr = *in;
	size_t szarr2 = sz1 * sz2 * sz3 * sz4;
	for (size_t i = 0; i < szarr2; i++) {
		arr2[i] = ptr;
		ptr += sz5;
	}
	return arr;
}

template <class type>
inline type***** newArr5_device(type** in, size_t sz1, size_t sz2, size_t sz3, size_t sz4, size_t sz5)
{
	type* inptr = newArr1_device<type>(sz1 * sz2 * sz3 * sz4 * sz5);
	cudaMemcpy(in, &inptr, sizeof(type*), cudaMemcpyHostToDevice);

	type***** arr = newArr4_device<type*>(sz1, sz2, sz3, sz4);
	type** arr2, *** arr3, **** arr4;
	cudaMemcpy(&arr4, arr, sizeof(type****), cudaMemcpyDeviceToHost);
	cudaMemcpy(&arr3, arr4, sizeof(type***), cudaMemcpyDeviceToHost);
	cudaMemcpy(&arr2, arr3, sizeof(type**), cudaMemcpyDeviceToHost);

	type* ptr = inptr;
	size_t szarr2 = sz1 * sz2 * sz3 * sz4;
	type* ptrarr[szarr2];
	for (size_t i = 0; i < szarr2; i++) {
		ptrarr[i] = ptr;
		ptr += sz5;
	}
	cudaMemcpy(arr2, ptrarr, sizeof(type*) * szarr2, cudaMemcpyHostToDevice);
	return arr;
}


template <class type>
inline type**** newArr4(type** in, size_t sz1, size_t sz2, size_t sz3, size_t sz4)
{
	*in = newArr1<type>(sz1 * sz2 * sz3 * sz4);

	type**** arr = newArr3<type*>(sz1, sz2, sz3);
	type** arr2 = **arr;
	type* ptr = *in;
	size_t szarr2 = sz1 * sz2 * sz3;
	for (size_t i = 0; i < szarr2; i++) {
		arr2[i] = ptr;
		ptr += sz4;
	}
	return arr;
}

template <class type>
inline type**** newArr4_device(type** in, size_t sz1, size_t sz2, size_t sz3, size_t sz4)
{
	type* inptr = newArr1_device<type>(sz1 * sz2 * sz3 * sz4);
	cudaMemcpy(in, &inptr, sizeof(type*), cudaMemcpyHostToDevice);

	type**** arr = newArr3_device<type*>(sz1, sz2, sz3);
	type** arr2, *** arr3;
	cudaMemcpy(&arr3, arr, sizeof(type***), cudaMemcpyDeviceToHost);
	cudaMemcpy(&arr2, arr3, sizeof(type**), cudaMemcpyDeviceToHost);

	type* ptr = inptr;
	size_t szarr2 = sz1 * sz2 * sz3;
	type* ptrarr[szarr2];
	for (size_t i = 0; i < szarr2; i++) {
		ptrarr[i] = ptr;
		ptr += sz4;
	}
	cudaMemcpy(arr2, ptrarr, sizeof(type*) * szarr2, cudaMemcpyHostToDevice);
	return arr;
}

template <class type>
inline type*** newArr3(type** in, size_t sz1, size_t sz2, size_t sz3)
{
	*in = newArr1<type>(sz1 * sz2 * sz3);

	type*** arr = newArr2<type*>(sz1, sz2);
	type** arr2 = *arr;
	type* ptr = *in;
	size_t szarr2 = sz1 * sz2;
	for (size_t i = 0; i < szarr2; i++) {
		arr2[i] = ptr;
		ptr += sz3;
	}
	return arr;
}

template <class type>
inline type*** newArr3_device(type** in, size_t sz1, size_t sz2, size_t sz3)
{
	type* inptr = newArr1_device<type>(sz1 * sz2 * sz3);
	cudaMemcpy(in, &inptr, sizeof(type*), cudaMemcpyHostToDevice);

	type*** arr = newArr2_device<type*>(sz1, sz2);
	type** arr2;
	cudaMemcpy(&arr2, arr, sizeof(type**), cudaMemcpyDeviceToHost);

	type* ptr = inptr;
	size_t szarr2 = sz1 * sz2;
	type* ptrarr[szarr2];
	for (size_t i = 0; i < szarr2; i++) {
		ptrarr[i] = ptr;
		ptr += sz3;
	}
	cudaMemcpy(arr2, ptrarr, sizeof(type*) * szarr2, cudaMemcpyHostToDevice);
	return arr;
}


template <class type>
inline type** newArr2(type** in, size_t sz1, size_t sz2)
{
	*in = newArr1<type>(sz1 * sz2);

	type** arr = newArr1<type*>(sz1);
	type* ptr = *in;
	for (size_t i = 0; i < sz1; i++) {
		arr[i] = ptr;
		ptr += sz2;
	}
	return arr;
}

template <class type>
inline type** newArr2_device(type** in, size_t sz1, size_t sz2)
{
	type* inptr = newArr1_device<type>(sz1 * sz2);
	cudaMemcpy(in, &inptr, sizeof(type*), cudaMemcpyHostToDevice);

	type** arr = newArr1_device<type*>(sz1);

	type* ptr = inptr;
	type* ptrarr[sz1];
	for (size_t i = 0; i < sz1; i++) {
		ptrarr[i] = ptr;
		ptr += sz2;
	}
	cudaMemcpy(arr, ptrarr, sizeof(type*) * sz1, cudaMemcpyHostToDevice);
	return arr;
}

// methods to deallocate arrays
//
template < class type >
inline void delArray1(type* arr)
{
	delete[](arr);
}

template < class type >
inline void delArray2(type** arr)
{
	delArray1(arr[0]); delete[](arr);
}

template < class type >
inline void delArray3(type*** arr)
{
	delArray2(arr[0]); delete[](arr);
}

template < class type >
inline void delArray4(type**** arr)
{
	delArray3(arr[0]); delete[](arr);
}

// device
template < class type >
inline void delArray1_device(type* arr)
{
	cudaFree(arr);
}

template < class type >
inline void delArray2_device(type** arr)
{
	type* arr1;
	cudaMemcpy(&arr1, arr, sizeof(type*), cudaMemcpyDeviceToHost); 
       	delArray1_device(arr1); cudaFree(arr);
}

template < class type >
inline void delArray3_device(type*** arr)
{
	type** arr2;
	cudaMemcpy(&arr2, arr, sizeof(type**), cudaMemcpyDeviceToHost);
       	delArray2_device(arr2); cudaFree(arr);
}

template < class type >
inline void delArray4_device(type**** arr)
{
	type*** arr3;
	cudaMemcpy(&arr3, arr, sizeof(type***), cudaMemcpyDeviceToHost);
	delArray3_device(arr3); cudaFree(arr);
}

//
// versions with dummy dimensions (for backwards compatibility)
//
template <class type>
inline void delArr1(type* arr)
{
	delArray1<type>(arr);
}

template <class type>
inline void delArr2(type** arr, size_t sz1)
{
	delArray2<type>(arr);
}

template <class type>
inline void delArr3(type*** arr, size_t sz1, size_t sz2)
{
	delArray3<type>(arr);
}

template <class type>
inline void delArr4(type**** arr, size_t sz1, size_t sz2, size_t sz3)
{
	delArray4<type>(arr);
}

#define newArr1(type, sz1) newArr1<type>(sz1)
#define newArr(type,sz1,sz2) newArr2<type>(sz1, sz2)
#define newArr2(type, sz1, sz2) newArr2<type>(sz1, sz2)
#define newArr3(type, sz1, sz2, sz3) newArr3<type>(sz1, sz2, sz3)
#define newArr4(type, sz1, sz2, sz3, sz4) newArr4<type>(sz1, sz2, sz3, sz4)

#endif
