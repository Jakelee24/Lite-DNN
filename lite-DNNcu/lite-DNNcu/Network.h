#pragma once

namespace LiteDNN {

class Network
{
public:
	Network();
	virtual ~Network();
public:
	float getLoss();
};

}

