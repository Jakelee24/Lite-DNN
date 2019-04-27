#pragma once
#include <memory>
namespace LiteDNN {
struct DataSize
{
public:
	DataSize() = default;
	DataSize( const size_t pNumber, const size_t pChannels, const size_t pWidth, const size_t pHeight )
		:mNumber( pNumber ), mChannels( pChannels ), mWidth( pWidth ), mHeight( pHeight ) {};
	inline size_t totalSize() const { return _4DSize(); }
	inline size_t _4DSize() const { return mNumber*mChannels*mWidth*mHeight; }
	inline size_t _3DSize() const { return mChannels*mWidth*mHeight; }
	inline size_t _2DSize() const { return mWidth*mHeight; }
	inline bool operator==( const DataSize& other )const {
		return other.mNumber == mNumber && other.mChannels == mChannels && other.mWidth == mWidth && other.mHeight == mHeight;
	}
	inline bool operator!=( const DataSize& other )const {
		return !( *this == ( other ) );
	}
	inline size_t getIndex( const size_t in, const size_t ic, const size_t ih, const size_t iw )const {
		return in*mChannels*mHeight*mWidth + ic*mHeight*mWidth + ih*mWidth + iw;
	}
	inline size_t getIndex( const size_t ic, const size_t ih, const size_t iw )const {
		return ic*mHeight*mWidth + ih*mWidth + iw;
	}
	size_t mNumber = 0;
	size_t mChannels = 0;
	size_t mWidth = 0;
	size_t mHeight = 0;
};
class DataFlow
{
public:
	DataFlow();
	DataFlow( const DataSize _size );
	virtual ~DataFlow();
public:
	DataFlow getSize() const;
	std::shared_ptr<float> getData() const;
	void fillData( const float item );
	void cloneTo( DataFlow& target );
private:
	DataSize size;
	std::shared_ptr<float> data;
};
}
