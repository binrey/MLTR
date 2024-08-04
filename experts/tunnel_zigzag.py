from .base import *


class ClsTunZigZag(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsTunZigZag, self).__init__(cfg, name="tuntrend")
        self.zigzag = ZigZagNew(self.cfg.period, 32)
        
    def __call__(self, common, h) -> bool:
        is_fig = False 
        zz_ids, zz_values, zz_types = self.zigzag.update(h)
        # zz_types, zz_values = np.array(zz_types), np.array(zz_values)
        # if len(zz_ids) < 3:
        #     return is_fig
        mid_line = sum(zz_values[-3:-1])/2
        mid_line_final = mid_line
        last_id = -4
        ncross, metric = 1, np.Inf
        for i in range(-4, -len(zz_ids)+1, -1):
            mid_line = sum(zz_values[i:-1])/(-1-i)
            new_cross = False
            new_cross += zz_types[i] > 0 and zz_values[i] >= mid_line and zz_values[i+1] < mid_line
            new_cross += zz_types[i] < 0 and zz_values[i] <= mid_line and zz_values[i+1] > mid_line

            if new_cross:
                ncross += 1
                mid_line_final = sum(zz_values[i:-1])/(-1-i)
                last_id = abs(i)
            # else:
            #     break
                
        peaks_upper = zz_values[-last_id+1:-1][zz_types[-last_id+1:-1]>0]
        peaks_bottom = zz_values[-last_id+1:-1][zz_types[-last_id+1:-1]<0]
        if len(peaks_upper) and len(peaks_bottom):
            lprice = peaks_upper.mean()
            sprice = peaks_bottom.mean()
        
            # metric = (zz_ids[-2] - zz_ids[-last_id]) / ((lprice - sprice) / mid_line_final) / 100
            metric = ncross
            if metric > self.cfg.ncross:
                is_fig = True


        if is_fig:
            common.lines = [[(x, y) for x, y in zip(zz_ids, zz_values)]]
            common.lprice = lprice
            common.sprice = sprice
            common.sl = {1: min(peaks_bottom), -1: max(peaks_upper)}  
            # common.tp = {1: tp, -1: tp}
            # common.cprice = {-1: min(zz_values[-last_id:]), 1: max(zz_values[-last_id:])} 
            common.lines += [[(zz_ids[-last_id], common.lprice), (h.Id[-1], common.lprice)],
                             [(zz_ids[-last_id], common.sprice), (h.Id[-1], common.sprice)],
                             [(zz_ids[-last_id], mid_line_final), (h.Id[-1], mid_line_final)]
                             ]
        return is_fig