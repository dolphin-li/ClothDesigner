#pragma once


enum QImdebugType
{
	QImdebugType_RGB,
	QImdebugType_RGBA,
	QImdebugType_GRAY,
	QImdebugType_INDEX,
	QImdebugType_INDEX_RGB,
	QImdebugType_INDEX_RGBA,
};


bool qimdebug(const char* name, const unsigned char* data, int w, int h, QImdebugType type, bool reverse_y = false);
bool qimdebug(const char* name, const float* data, int w, int h, QImdebugType type, bool reverse_y = false);
bool qimdebug(const char* name, const int* data, int w, int h, QImdebugType type, bool reverse_y = false);




