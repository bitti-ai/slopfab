#include "internal.h"
#include "slopfab/cuda/device.h"
#include "slopfab/cuda/gemm.cuh"
#include <algorithm>
#include <cmath>

namespace slopfab::whisper {
namespace {
using Buffer=cuda::DeviceBuffer<float>;
float* ptr(const Tensor& t){return std::static_pointer_cast<Buffer>(t.data)->get();}
Tensor make(int rows,int cols){return {rows,cols,std::make_shared<Buffer>(size_t(rows)*cols)};}
__global__ void pointwise(const float* x,const float* w,float* y,int n,int cols,int op) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  if(op==0)y[i]=x[i]+w[i];
  else if(op==1)y[i]=0.5f*x[i]*(1+erff(x[i]*0.7071067811865475f));
  else y[i]=x[i]+w[i%cols];
}
__global__ void normalize(const float* x,const float* w,const float* b,float* y,int cols) {
  __shared__ float s[256];int t=threadIdx.x,r=blockIdx.x;
  float sum=0;for(int c=t;c<cols;c+=256)sum+=x[size_t(r)*cols+c];
  s[t]=sum;__syncthreads();
  for(int d=128;d;d/=2){if(t<d)s[t]+=s[t+d];__syncthreads();}
  float mean=s[0]/cols;__syncthreads();
  sum=0;for(int c=t;c<cols;c+=256){float a=x[size_t(r)*cols+c]-mean;sum+=a*a;}
  s[t]=sum;__syncthreads();
  for(int d=128;d;d/=2){if(t<d)s[t]+=s[t+d];__syncthreads();}
  float inv=rsqrtf(s[0]/cols+1e-5f);
  for(int c=t;c<cols;c+=256)y[size_t(r)*cols+c]=(x[size_t(r)*cols+c]-mean)*inv*w[c]+b[c];
}
__global__ void columns(const float* x,float* y,int rows,int cols,int outrows,int stride) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=outrows*cols*3)return;
  int r=i/(cols*3),c=(i/3)%cols,k=i%3,at=r*stride+k-1;
  y[i]=at>=0 && at<rows?x[size_t(at)*cols+c]:0;
}
__global__ void softmax(float* x,int keys,bool causal) {
  __shared__ float s[256];int t=threadIdx.x,r=blockIdx.x;
  int limit=causal?min(keys,r+1):keys;float mx=-INFINITY;
  for(int k=t;k<limit;k+=256)mx=fmaxf(mx,x[size_t(r)*keys+k]);
  s[t]=mx;__syncthreads();for(int d=128;d;d/=2){if(t<d)s[t]=fmaxf(s[t],s[t+d]);__syncthreads();}
  mx=s[0];__syncthreads();float sum=0;
  for(int k=t;k<limit;k+=256)sum+=expf(x[size_t(r)*keys+k]-mx);
  s[t]=sum;__syncthreads();for(int d=128;d;d/=2){if(t<d)s[t]+=s[t+d];__syncthreads();}
  sum=s[0];for(int k=t;k<keys;k+=256)x[size_t(r)*keys+k]=k<limit?expf(x[size_t(r)*keys+k]-mx)/sum:0;
}
class CUDA final:public Backend {
  cublasHandle_t blas=nullptr;
public:
  CUDA(){SLOPFAB_CUBLAS_CHECK(cuda::cublas_create(&blas));}
  ~CUDA(){if(blas)cuda::cublas_destroy(blas);}
  Tensor upload(const std::vector<float>& v,int r,int c)override {
    if(v.size()!=size_t(r)*c)throw std::invalid_argument("Whisper: upload shape");
    auto t=make(r,c);SLOPFAB_CUDA_CHECK(cudaMemcpy(ptr(t),v.data(),v.size()*4,cudaMemcpyHostToDevice));return t;
  }
  std::vector<float> download(const Tensor& t)override {
    std::vector<float> v(size_t(t.rows)*t.cols);SLOPFAB_CUDA_CHECK(cudaMemcpy(v.data(),ptr(t),v.size()*4,cudaMemcpyDeviceToHost));return v;
  }
  Tensor linear(const Tensor& x,const Tensor& w,const Tensor* b)override {
    auto y=make(x.rows,w.rows);float one=1,zero=0;
    SLOPFAB_CUBLAS_CHECK(cuda::cublas_gemm_ex(blas,CUBLAS_OP_T,CUBLAS_OP_N,w.rows,x.rows,x.cols,
      &one,ptr(w),CUDA_R_32F,x.cols,ptr(x),CUDA_R_32F,x.cols,&zero,ptr(y),CUDA_R_32F,y.cols,
      CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
    if(b)pointwise<<<(y.rows*y.cols+255)/256,256>>>(ptr(y),ptr(*b),ptr(y),y.rows*y.cols,y.cols,2);
    return y;
  }
  Tensor norm(const Tensor& x,const Tensor& w,const Tensor& b)override {
    auto y=make(x.rows,x.cols);normalize<<<x.rows,256>>>(ptr(x),ptr(w),ptr(b),ptr(y),x.cols);return y;
  }
  Tensor gelu(const Tensor& x)override {
    auto y=make(x.rows,x.cols);pointwise<<<(x.rows*x.cols+255)/256,256>>>(ptr(x),ptr(x),ptr(y),x.rows*x.cols,x.cols,1);return y;
  }
  Tensor add(const Tensor& x,const Tensor& w)override {
    auto y=make(x.rows,x.cols);pointwise<<<(x.rows*x.cols+255)/256,256>>>(ptr(x),ptr(w),ptr(y),x.rows*x.cols,x.cols,0);return y;
  }
  Tensor rows(const Tensor& x,int first,int count)override {
    auto y=make(count,x.cols);SLOPFAB_CUDA_CHECK(cudaMemcpy(ptr(y),ptr(x)+size_t(first)*x.cols,size_t(count)*x.cols*4,cudaMemcpyDeviceToDevice));return y;
  }
  Tensor concat(const Tensor& a,const Tensor& b)override {
    if(!a.data)return b;
    auto y=make(a.rows+b.rows,a.cols);
    SLOPFAB_CUDA_CHECK(cudaMemcpy(ptr(y),ptr(a),size_t(a.rows)*a.cols*4,cudaMemcpyDeviceToDevice));
    SLOPFAB_CUDA_CHECK(cudaMemcpy(ptr(y)+size_t(a.rows)*a.cols,ptr(b),size_t(b.rows)*b.cols*4,cudaMemcpyDeviceToDevice));return y;
  }
  Tensor im2col(const Tensor& x,int stride)override {
    auto y=make((x.rows+stride-1)/stride,x.cols*3);
    columns<<<(y.rows*y.cols+255)/256,256>>>(ptr(x),ptr(y),x.rows,x.cols,y.rows,stride);return y;
  }
  Tensor attention(const Tensor& q,const Tensor& k,const Tensor& v,int heads,bool causal,std::vector<float>* probabilities)override {
    auto y=make(q.rows,q.cols);auto scores=make(q.rows,k.rows);
    int dim=q.cols/heads;float scale=1/std::sqrt(float(dim)),one=1,zero=0;
    if(probabilities)probabilities->resize(size_t(heads)*q.rows*k.rows);
    for(int h=0;h<heads;++h) {
      SLOPFAB_CUBLAS_CHECK(cuda::cublas_gemm_ex(blas,CUBLAS_OP_T,CUBLAS_OP_N,k.rows,q.rows,dim,
        &scale,ptr(k)+h*dim,CUDA_R_32F,k.cols,ptr(q)+h*dim,CUDA_R_32F,q.cols,&zero,ptr(scores),CUDA_R_32F,k.rows,
        CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
      softmax<<<q.rows,256>>>(ptr(scores),k.rows,causal);
      if(probabilities)SLOPFAB_CUDA_CHECK(cudaMemcpy(probabilities->data()+size_t(h)*q.rows*k.rows,ptr(scores),size_t(q.rows)*k.rows*4,cudaMemcpyDeviceToHost));
      SLOPFAB_CUBLAS_CHECK(cuda::cublas_gemm_ex(blas,CUBLAS_OP_N,CUBLAS_OP_N,dim,q.rows,k.rows,
        &one,ptr(v)+h*dim,CUDA_R_32F,v.cols,ptr(scores),CUDA_R_32F,k.rows,&zero,ptr(y)+h*dim,CUDA_R_32F,y.cols,
        CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
    }
    return y;
  }
};
}
std::unique_ptr<Backend> cuda_backend(){return std::make_unique<CUDA>();}
}
