import numpy as np
from FPGANode import FBoltAsync
from CLFPGA import FBufferDescriptor, FBufferType


# import storm


# class VecSumBolt(storm.BasicBolt):

#     def process(self, tup):
#         A = tup.values[0]
#         B = tup.values[1]
#         C = [a + b for (a, b) in zip(A, B)]
#         storm.emit([C])


# VecSumBolt().run()


class VecSumBolt(FBoltAsync):

    def prepare_compute(self, tup):
        args = []
        args.append(None)
        args.append(np.asarray(tup.values[0], dtype=np.int32))
        args.append(np.asarray(tup.values[1], dtype=np.int32))
        args.append(np.int32(len(tup.values[0])))
        return args

    def prepare_emit(self, tup, results):
        size = len(tup.values[0])
        return [results[0][0:size], tup.values[2]]


xclbin_filepath = "vecsum_local.xclbin"
kernel_name = "vecsum"
vec_size = 8 * 1024
degree = 2
buff_descr = []
buff_descr.append(FBufferDescriptor(FBufferType.OUT, np.int32, "C", vec_size))
buff_descr.append(FBufferDescriptor(FBufferType.IN, np.int32, "A", vec_size))
buff_descr.append(FBufferDescriptor(FBufferType.IN, np.int32, "B", vec_size))
buff_descr.append(FBufferDescriptor(FBufferType.SCALAR, np.int32, "size", 1))
bolt = VecSumBolt(xclbin_filepath,
                  kernel_name,
                  buff_descr,
                  degree)
bolt.run()
