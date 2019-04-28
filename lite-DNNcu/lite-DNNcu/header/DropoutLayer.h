#pragma once
#include "Layer.h"

namespace LiteDNN {
	
	class DropoutLayer : public Layer
	{
		FRIEND_WITH_NETWORK
	public:
		DropoutLayer();
		DropoutLayer(const float _rate);
		virtual ~DropoutLayer();
	public:
		void setParameters(const float _rate);
	protected:
		DECLARE_LAYER_TYPE;
		virtual std::string serializeToString() const override;
		virtual void serializeFromString(const std::string content) override;
		virtual std::string getLayerType() const override;
		virtual void solveInnerParams() override;
		virtual void forward(const std::shared_ptr<DataFlow> prev, std::shared_ptr<DataFlow> next) override;
		virtual void backward(
			std::shared_ptr<DataFlow> prev, 
			const std::shared_ptr<DataFlow> next,
			std::shared_ptr<DataFlow>& prevDiff, 
			const std::shared_ptr<DataFlow>& nextDiff) override;
	private:
		float rate = 0.5f;
		std::shared_ptr<ParamData> mask;

	};

}