#ifndef DSA_EMULATE_H
#define DSA_EMULATE_H

/* dsa_emulate.h - support EXT_direct_state_access for GLEW if unsupported */

#ifdef __cplusplus
extern "C" {
#endif

extern void emulate_dsa_if_needed(int forceDSAemulation);

#ifdef __cplusplus
}
#endif

#endif /* DSA_EMULATE_H */
