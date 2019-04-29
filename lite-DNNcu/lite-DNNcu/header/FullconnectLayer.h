#pragma once
#include "Layer.h"

namespace LiteDNN {

	class FullconnectLayer : public Layer
	{
		FRIEND_WITH_NETWORK
	public:
		FullconnectLayer();
		virtual ~FullconnectLayer();
	public:
		void setParameters(const ParamSize _outMapSize, const bool _enabledBias);
	protected:
		DECLARE_LAYER_TYPE;
		virtual std::string serializeToString() const override;
		virtual void serializeFromString(const std::string content) override;
		virtual std::string getLayerType() const override;
		virtual void forward(const std::shared_ptr<DataFlow> prev,
			std::shared_ptr<DataFlow> next) override;

		virtual void backward(
			std::shared_ptr<DataFlow> prev,
			const std::shared_ptr<DataFlow> next,
			std::shared_ptr<DataFlow>& prevDiff, 
			const std::shared_ptr<DataFlow>& nextDiff
			)override;

	private:
		ParamSize outMapSize;
		std::shared_ptr<ParamData> weight;
		std::shared_ptr<ParamData> weightGradient;
		bool enableBias = false;
		std::shared_ptr<ParamData> bias;
		std::shared_ptr<ParamData> biasGradient;
	};

}