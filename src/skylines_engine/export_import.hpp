#ifndef SKYLINES_EXPORT_IMPORT_H_
#define SKYLINES_EXPORT_IMPORT_H_
#ifdef __cplusplus
#pragma warning(disable:4251)
#endif

#ifdef WIN32
#ifdef skylines_EXPORTS
#define skylines_DLL_EXPORTS __declspec(dllexport)
#else
#define skylines_DLL_EXPORTS __declspec(dllimport)
#endif
#else
#define skylines_DLL_EXPORTS
#endif

#endif // SKYLINES_EXPORT_IMPORT_H_
