#pragma once
#include "DataFlow.h"
#include "ParamData.h"

namespace LiteDNN {
	class LossFunction
	{
	public:
		virtual float getLoss(const std::shared_ptr<DataFlow> labelDataFlow,
			const std::shared_ptr<DataFlow> outputDataFlow) = 0;
		virtual void getDiff(const std::shared_ptr<DataFlow> labelDataFlow,
			const std::shared_ptr<DataFlow> outputDataFlow, std::shared_ptr<DataFlow> &diff) = 0;
	};

	class CrossEntropyFunctor : public LossFunction
	{
	public:
		virtual float getLoss(const std::shared_ptr<DataFlow> labelDataFlow,
			const std::shared_ptr<DataFlow> outputDataFlow);
		virtual void getDiff(const std::shared_ptr<DataFlow> labelDataFlow,
			const std::shared_ptr<DataFlow> outputDataFlow, std::shared_ptr<DataFlow>& diff);
	};

	class MSEFunctor : public LossFunction
	{
	public:
		virtual float getLoss(const std::shared_ptr<DataFlow> labelDataBucket,
			const std::shared_ptr<DataFlow> outputDataBucket);
		virtual void getDiff(const std::shared_ptr<DataFlow> labelDataBucket,
			const std::shared_ptr<DataFlow> outputDataBucket, std::shared_ptr<DataFlow>& diff);
	};
}