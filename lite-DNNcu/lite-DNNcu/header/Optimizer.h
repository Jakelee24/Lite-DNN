#pragma once
#include <vector>
#include "DataFlow.h"
#include "ParamData.h"

namespace LiteDNN {

	class Optimizer
	{
	public:
		Optimizer(const float _lr) :lr(_lr) {}
		void setLearningRate(const float _lr) { lr = _lr; }
		virtual void update(std::vector<std::shared_ptr<DataFlow>> params,
			const std::vector<std::shared_ptr<DataFlow>> gradients) = 0;
	protected:
		float lr = 0.0f;
	};

	class SGD : public Optimizer
	{

	};
}