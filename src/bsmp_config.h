#ifndef BSMP_CONFIG_H
#define BSMP_CONFIG_H

#ifndef BSMP_BLOCK_SIZE
#define BSMP_BLOCK_SIZE 4
#endif

namespace bsmp {
static constexpr int kBlockSize = BSMP_BLOCK_SIZE;
}

#endif // BSMP_CONFIG_H