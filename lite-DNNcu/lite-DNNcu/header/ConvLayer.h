#pragma once
#include "Layer.h"

namespace LiteDNN {

	class ConvLayer : public Layer
	{
		FRIEND_WITH_NETWORK
	public:
		enum PaddingType
		{
			VALID = 0,
			SAME = 1
		};
	public:
		ConvLayer();
		virtual ~ConvLayer();
		void setParameters(const ParamSize _kernelSize, const size_t _widthStep, const size_t _heightStep,
			const bool _enabledBias, const PaddingType _paddingType);
	
	protected:
		DECLARE_LAYER_TYPE;
		virtual std::string serializeToString() const override;

		virtual void serializeFromString(const std::string content) override;

		virtual std::string getLayerType() const override;

		virtual void solveInnerParams() override;

		virtual void forward(
			const std::shared_ptr<DataFlow> prev,
			std::shared_ptr<DataFlow> next) override;

		virtual void backward(
			std::shared_ptr<DataFlow> prev,
			const std::shared_ptr<DataFlow> next, 
			std::shared_ptr<DataFlow>& prevDiff, 
			const std::shared_ptr<DataFlow>& nextDiff) override;

	private:
		ParamSize kernelSize;
		size_t widthStep = 0;
		size_t heightStep = 0;
		std::shared_ptr<ParamData> kernel;
		std::shared_ptr<ParamData> kernelGradient;
		bool enabledBias = false;
		PaddingType paddingType = VALID;
		std::shared_ptr<ParamData> bias;
		std::shared_ptr<ParamData> biasGradient;
	};

}