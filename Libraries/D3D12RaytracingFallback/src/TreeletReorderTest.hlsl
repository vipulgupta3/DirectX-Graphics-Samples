//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

// Using the Karras/Aila paper on treelet reoordering:
// "Fast Parallel Construction of High-Quality Bounding Volume
// Hierarchies"

#define HLSL
#include "TreeletReorderBindings.h"
#include "RayTracingHelper.hlsli"

// Must be at least FullTreeletSize
#define NumThreadsInGroup 32

[numthreads(NumThreadsInGroup, 1, 1)]
void main(uint3 Gid : SV_GroupID, uint3 GTid : SV_GroupThreadId)
{
    uint threadId = GTid.x;

    if (threadId != 0)
    {
        return;
    }

    uint numBitmasksPerThread;
    uint extraBitmasks;
    uint bitmasksStart;
    uint bitmasksEnd;

    uint debugCursor = 0;

    uint subsetSize = 3;
    // eg. In 'treelet/subset' of size 2, there are (7 Choose 2) distinct 'treelets' in the original treelet of 7 leaves, ie. 0000011, 0000101, ..., 1100000
    uint numTreeletBitmasks = FullTreeletSizeChoose[subsetSize];
    numBitmasksPerThread = max(numTreeletBitmasks / NumThreadsInGroup, 1);
    extraBitmasks = numTreeletBitmasks > NumThreadsInGroup ? numTreeletBitmasks % NumThreadsInGroup : 0;

    bitmasksStart = numBitmasksPerThread * threadId + min(threadId, extraBitmasks);
    bitmasksEnd = bitmasksStart + numBitmasksPerThread + (threadId < extraBitmasks ? 1 : 0);

    uint treeletBitmask = GetBitPermutation(subsetSize, 0);

    uint delta = (treeletBitmask - 1) & treeletBitmask;
    uint partitionBitmask = (-delta) & treeletBitmask;

    ASDebugLog[debugCursor++] = partitionBitmask;

    do
    {
        partitionBitmask = (partitionBitmask - delta) & treeletBitmask;

        ASDebugLog[debugCursor++] = partitionBitmask;
        if (debugCursor >= AccelerationStructureDebugLogLength) return;
    } while (partitionBitmask != 0);
}