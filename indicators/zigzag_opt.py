import numpy as np
from easydict import EasyDict
from loguru import logger

from .zigzag import ZigZag


class ZigZagOpt(ZigZag):
    def __init__(self, period=1, min_nodes=1, max_drop=1, simp_while_grow=False):
        super(ZigZagOpt, self).__init__(period, out_size=32)
        self.min_nodes = min_nodes
        self.simp_while_grow = simp_while_grow
        self.max_drop = max_drop
        self.masks = []
    
    def zigzag_simplify(self, data, mask, only_calc=False):
        def swap_node(data, node):
            data[node.id1:node.id2+1] = -data[node.id1:node.id2+1]
            return data
        smax, nnodes = 0, 1
        node = EasyDict({"value": mask[0], "id1": 0, "id2": 0, "metric":None})
        nodemax = node
        for i in range(1, mask.shape[0]):
            if mask[i] == node.value:
                node.id2 = i
            else:
                if not only_calc:
                    mask = swap_node(mask, node)
                    s = (data*mask).sum()
                    if s >= smax:
                        # logger.debug(f"{i}, {nnodes}, {node}, {s:+5.3f}")
                        smax = s
                        nodemax = node
                        nodemax.metric = s
                        # logger.debug("OK")
                    mask = swap_node(mask, node)
                    # print("")
                node = EasyDict({"value": mask[i], "id1": i, "id2": i, "metric":None})
                nnodes += 1
        if not only_calc:
            mask = swap_node(mask, nodemax)
            nnodes -= 1
        else:
            return mask, nnodes, (data*mask).sum()
        if nodemax.metric is None:
            nodemax.metric = (data*mask).sum()
        return mask, nnodes, nodemax.metric


    def update(self, h):
        hclose = h.Close
        dh = hclose[1:] - hclose[:-1]
        mask = self._get_mask(h)
        i, v, t = self.mask2zigzag(h, mask, True) 
        _, nnodes, res = self.zigzag_simplify(dh, mask, only_calc=True)
        logger.debug(f"{nnodes:003}, {res:>7.2f}, {0}")    
        self.res_list = [res]
        self.nnodes_list = [nnodes]
        res_max = res
        while nnodes > self.min_nodes:
            mask_, nnodes, res = self.zigzag_simplify(dh, mask.copy())
            drop = (res_max - res)/max(1, res_max)
            logger.debug(f"{nnodes:003}, {res:>7.2f}, {drop}")    
                
            
            if res < self.res_list[-1]:
                if self.simp_while_grow:
                    break
            else:
                res_max = res
            if drop > self.max_drop:
                break
            self.res_list.append(res)
            self.nnodes_list.append(nnodes)
            mask = mask_.copy()
            self.masks.append(mask)
        return self.mask2zigzag(h, mask, True)
    
    def plot(self, save_path):
        import matplotlib.pylab as plt
        plt.plot(self.nnodes_list, self.res_list, ".-")
        plt.savefig(save_path / "zz_opt.jpg")



    

        
