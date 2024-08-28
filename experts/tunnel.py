from experts.base import ExtensionBase


class ClsTunnel(ExtensionBase):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ClsTunnel, self).__init__(cfg, name="tunnel")

    def __call__(self, common, h) -> bool:
        is_fig = False
        best_params = {
            "metric": 0,
            "i": 0,
            "line_above": None,
            "line_below": None,
        }
        for i in range(4, h.Id.shape[0], 1):
            line_above = h.High[-i:].mean()
            line_below = h.Low[-i:].mean()
            middle_line = (line_above + line_below) / 2

            if h.Close[-1] < line_above and h.Close[-1] > line_below:
                metric = i / ((line_above - line_below) / middle_line) / 100

                if metric > best_params["metric"]:
                    best_params.update(
                        {"metric": metric,
                        "i": i,
                        "line_above": line_above,
                        "line_below": line_below,
                        "middle_line": middle_line
                        }
                    )

        if best_params["metric"] > self.cfg.ncross:
            is_fig = True
            # break

        if is_fig:
            i = best_params["i"]
            common.sl = {1: h.Low[-i:].min(), -1: h.High[-i:].max()}
            # v1
            common.lprice = best_params["line_above"]
            common.sprice = best_params["line_below"]
            # v2
            # if middle_line > h.Close.mean():
            #     common.lprice = line_below
            # else:
            #     common.sprice = line_above
            common.lines = [[(h.Id[-i], best_params["line_above"]), (h.Id[-1], best_params["line_above"])],
                            [(h.Id[-i], best_params["line_below"]), (h.Id[-1], best_params["line_below"])],
                            [(h.Id[-i], best_params["middle_line"]), (h.Id[-1], best_params["middle_line"])]]

        return is_fig