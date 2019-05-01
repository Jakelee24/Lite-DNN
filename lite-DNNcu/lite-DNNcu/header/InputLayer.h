#pragma once
#include "Layer.h"

namespace LiteDNN {
	class InputLayer : public Layer
	{
		FRIEND_WITH_NETWORK
	public:
		InputLayer();
		virtual ~InputLayer();
	protected:
		DECLARE_LAYER_TYPE;
		virtual std::string serializeToString() const override;
		virtual void serializeFromString(const std::string content) override;
		virtual std::string getLayerType() const override;
		virtual void forward(const std::shared_ptr<DataFlow> prev, 
			std::shared_ptr<DataFlow> next) override;
		virtual void backward(std::shared_ptr<DataFlow> prev, 
			const std::shared_ptr<DataFlow> next,
			std::shared_ptr<DataFlow>& prevDiff,
			const std::shared_ptr<DataFlow>& nextDiff) override;
	};
}