#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_SIZE = 32;

template<typename TYPE_X,typename TYPE_Y,typename TYPE_Z> class KernelSArg {
public:
    __aicore__ inline KernelSArg() {}
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR indice,GM_ADDR values,uint32_t lx,uint32_t ly,uint32_t lz,
                                uint32_t dim,uint32_t dimNum,uint32_t tileDataNum,uint32_t tileNum,uint32_t tailLength,uint32_t stride/* 开发者填充参数列表 */)
    {
        //考生补充初始化代码
        ASSERT(GetBlockNum()!=0&&"block dim can not be zero!");
        this->lx=lx;
        this->ly=ly;
        this->lz=lz;
        int32_t blockNum=GetBlockNum();
        int32_t blockId=GetBlockIdx();
        printf("blockNum:%d\tblockId:%d\n",blockNum,blockId);
        printf("x:%d\ty:%d\tz:%d\n",lx,ly,lz);
        this->dim=dim;
        this->dimNum=dimNum;
        this->dataNum=lx*ly*lz;
        printf("tileDataNum:%d\ttileNum:%d\ttailLength:%d\n",tileDataNum,tileNum,tailLength);
        printf("dim:%d\tdimNum:%d\tdataNum:%d\n",dim,dimNum,dataNum);
        this->tileDataNum=tileDataNum;
        this->tileBlockNum=tileDataNum/(BLOCK_SIZE/sizeof(TYPE_X));
        this->tileNum=tileNum;
        this->tailLength=tailLength;
        this->stride=stride;
        // ASSERT(tileNum!=0&&"tile num can not be zero!");
        // this->tileLength=this->blockLength/tileNum/BUFFER_NUM;

        //用多核优化时才考虑idx
        // xGm.SetGlobalBuffer((__gm__ TYPE_X*)x+this->blockLength*GetBlockIdx(),this->blockLength);
        // yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y+this->blockLength*GetBlockIdx(),this->blockLength);
        
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x,this->dataNum);
        if(this->dim==0){
            indiceGm.SetGlobalBuffer((__gm__ TYPE_Y*)indice,this->dataNum/this->lx);
            valuesGm.SetGlobalBuffer((__gm__ TYPE_Z*)values,this->dataNum/this->lx);
        }
        else if(this->dim==1){
            indiceGm.SetGlobalBuffer((__gm__ TYPE_Y*)indice,this->dataNum/this->ly);
            valuesGm.SetGlobalBuffer((__gm__ TYPE_Z*)values,this->dataNum/this->ly);
        }
        else{
            indiceGm.SetGlobalBuffer((__gm__ TYPE_Y*)indice,this->dataNum/this->lz);
            valuesGm.SetGlobalBuffer((__gm__ TYPE_Z*)values,this->dataNum/this->lz);
        }


        pipe.InitBuffer(inQueueX,BUFFER_NUM,this->tileDataNum*sizeof(TYPE_X));
        pipe.InitBuffer(outQueueIndice,BUFFER_NUM,this->tileDataNum*sizeof(TYPE_Y));
        pipe.InitBuffer(outQueueValues,BUFFER_NUM,this->tileDataNum*sizeof(TYPE_Z));
        pipe.InitBuffer(tmpBuffer1,this->tileDataNum*sizeof(TYPE_X));
        pipe.InitBuffer(tmpBuffer2,this->tileDataNum*sizeof(TYPE_X));
        pipe.InitBuffer(tmpBuffer3,this->tileDataNum*sizeof(TYPE_X));
        pipe.InitBuffer(tmpBuffer4,this->tileDataNum*sizeof(TYPE_X));
        
    }
    __aicore__ inline void Process()
    {
        //考生补充对“loopCount”的定义，注意对Tiling的处理
        //大的循环：总数/每次规约的个数
        // int32_t groupNum=this->dim==0?1:BLOCK_SIZE/sizeof(TYPE_X);
        // int32_t leftT=this->lx*this->ly*this->lz/this->dimNum;//64
        // int32_t loopCount1=leftT/groupNum;//64
        // int32_t lastGroup=leftT%groupNum;
        // loopCount1=lastGroup==0?loopCount1:loopCount1+1;
        // int32_t loopCount2=this->tileNum;
        int32_t loopCount=this->tileNum;
        this->processDataNum=this->tileDataNum;


        for(int32_t i=0;i<loopCount;i++){
            if(i==loopCount-1&&this->tailLength!=0){
                this->processDataNum=this->tailLength;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //考生补充算子代码
        LocalTensor<TYPE_X> xLocal=inQueueX.AllocTensor<TYPE_X>();
        printf("process:%d\tprocessDataNum:%d\n",progress,this->processDataNum);
        DataCopy(xLocal,xGm,this->processDataNum);
        // if(this->dim==0){
            
        //     DataCopy(xLocal,xGm[progress*this->lx+tileProgress*this->tileDataNum],this->processDataNum);
        // }
        // else if(this->dim==1){
        //     uint32_t zNum=progress/(BLOCK_SIZE/sizeof(TYPE_X));
        //     uint32_t baseNum=progress%(BLOCK_SIZE/sizeof(TYPE_X));
        //     DataCopy(xLocal,xGm[progress*(zNum*this->lx*this->ly+baseNum*(BLOCK_SIZE/sizeof(TYPE_X)))+tileProgress*this->tileBlockNum*this->lx],{1,tileBlockNum,ly-1,0});//步长        
        // }
        // else{
        //     DataCopy(xLocal,xGm[progress*(BLOCK_SIZE/sizeof(TYPE_X))+tileProgress*this->lx*this->ly*this->tileBlockNum],{1,tileBlockNum,lx*ly-1,0});

        // }
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        //考生补充算子计算代码
        LocalTensor<TYPE_X> xLocal=inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> indiceLocal=outQueueIndice.AllocTensor<TYPE_Y>();
        LocalTensor<TYPE_Z> valuesLocal=outQueueValues.AllocTensor<TYPE_Z>();
        // LocalTensor<TYPE_X> tmpTensor1=tmpBuffer1.Get<TYPE_X>();
        // LocalTensor<TYPE_X> tmpTensor2=tmpBuffer2.Get<TYPE_X>();
        // LocalTensor<TYPE_X> tmpTensor3=tmpBuffer3.Get<TYPE_X>();
        // LocalTensor<TYPE_X> tmpTensor4=tmpBuffer4.Get<TYPE_X>();
        
       
        DumpTensor(xLocal,0,processDataNum);
        outQueueIndice.EnQue<TYPE_Y>(indiceLocal);
        outQueueValues.EnQue<TYPE_Z>(valuesLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        //考生补充算子代码
        LocalTensor<TYPE_Y>indiceLocal=outQueueIndice.DeQue<TYPE_Y>();
        LocalTensor<TYPE_Z>valuesLocal=outQueueValues.DeQue<TYPE_Z>();
        // DataCopy(indiceGm[progress*this->tileDataNum],indiceLocal,this->tileDataNum);
        // DataCopy(valuesGm[progress*this->tileDataNum],valuesLocal,this->tileDataNum);
        outQueueIndice.FreeTensor(indiceLocal);
        outQueueValues.FreeTensor(valuesLocal);
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueIndice,outQueueValues;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_Y> indiceGm;
    GlobalTensor<TYPE_Z> valuesGm;
    //考生补充自定义成员变量
    TBuf<QuePosition::VECCALC> tmpBuffer1,tmpBuffer2,tmpBuffer3,tmpBuffer4;

    uint32_t lx;
    uint32_t ly;
    uint32_t lz;

    uint32_t dataNum;   //数据总数
    uint32_t dim;
    uint32_t dimNum;    //目标维数上的长度
    uint32_t tileDataNum;
    uint32_t tileBlockNum;
    uint32_t tileNum;
    uint32_t tailLength;
    uint32_t stride;
    uint32_t processDataNum;
    // uint32_t tileNum;
    // uint32_t tileLength;
};
extern "C" __global__ __aicore__ void arg_max_with_value(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelSArg<DTYPE_X, DTYPE_INDICE, DTYPE_VALUES> op;
    op.Init(x,indice,values,tiling_data.lx,tiling_data.ly,tiling_data.lz,
    tiling_data.dim,tiling_data.dimNum,tiling_data.tileDataNum,tiling_data.tileNum,tiling_data.tailLength,tiling_data.stride);
    op.Process();
}