from abc import ABC, abstractmethod


class CandleFig(ABC):
    def __init__(self, min_history_size):
        self.min_history_size = min_history_size
        self.body_length = None
        self.trend_length = None
        self.target_length = None
        self.lines = None
            
    @abstractmethod
    def get_body(self) -> None:
        self.body_length = None
        self.body_line = None

    @abstractmethod
    def get_trend(self) -> None:
        self.trend_line = None
        
    @abstractmethod
    def get_prediction(self) -> None:
        self.prediction = None

    @abstractmethod
    def get_target(self) -> bool:
        self.target_line = None
        self.target = None
        self.target_length = None
        
    def update(self, h, t):
        assert h.shape[0] > self.min_history_size
        if self.get_body(h[:t]):
            self.get_trend(h[:t-self.body_length+1])
            self.get_prediction(h[:t])
            self.get_target(h[t:])
            self.lines = [self.trend_line, self.body_line, self.target_line]
            return True
        return False


class DenseLine(CandleFig):
    def __init__(self, body_maxsize, trend_maxsize, n_intersections, target_length):
        self.body_maxsize = body_maxsize
        self.trend_maxsize = trend_maxsize  
        self.n_intersections = n_intersections
        super(DenseLine, self).__init__(self.body_maxsize + self.trend_maxsize)
        self.center_line = None     
        self.target_length = target_length 
    
    def check_line(self, h, line):
        n, tlast = 0, 0
        for t in range(self.body_maxsize):
            if max(h.Close[-t], h.Open[-t]) > line and min(h.Close[-t], h.Open[-t]) < line:
                n += 1
                tlast = t
        return n, tlast
    
    def get_body(self, h):
        self.line = None
        line = h.Close[-1]
        n, t = self.check_line(h, line)
        if n >= self.n_intersections:
            self.body_length = t - 1
            self.center_line = line
            self.body_line = [(h.index[-self.body_length], line), (h.index[-1], line)]
            return True
        return False
    
    def get_trend(self, h) -> bool:
        self.trend_line = [(), (h.index[-1], self.center_line)]
        tmin = -self.trend_maxsize + h.Low[-self.trend_maxsize:].argmin()
        tmax = -self.trend_maxsize + h.High[-self.trend_maxsize:].argmax()
        if h.High[tmax] - self.center_line > self.center_line - h.Low[tmin]:
            self.trend_line[0] = (h.index[tmax], h.High[tmax])
            self.trend_length = -tmax
        else:
            self.trend_line[0] = (h.index[tmin], h.Low[tmin])    
            self.trend_length = -tmin           
    
    def get_prediction(self, h):
        self.prediction = 1 if self.trend_line[1][1] > self.trend_line[0][1] else -1
    
    def get_target(self, h):
        self.target_length = int(self.body_length*self.target_length)
        ptrg = h.Close.values[self.target_length]
        self.target = (ptrg - h.Close.values[0])/h.Close.values[0]
        self.target_line = [(h.index[0], self.center_line), (h.index[self.target_length], ptrg)]
        
        
class Triangle(CandleFig):
    def __init__(self, body_maxsize, trend_maxsize, n_intersections):
        self.body_maxsize = body_maxsize
        self.trend_maxsize = trend_maxsize  
        self.n_intersections = n_intersections       
        super(Triangle, self).__init__(self.body_maxsize + self.trend_maxsize)
        self.center_line = None      
    
    def check_line(self, h, line):
        n, tlast = 0, 0
        for t in range(self.body_maxsize):
            if max(h.Close[-t], h.Open[-t]) > line and min(h.Close[-t], h.Open[-t]) < line:
                n += 1
                tlast = t
        return n, tlast
    
    def get_body(self, h):
        self.line = None
        line = h.Close[-1]
        n, t = self.check_line(h, line)
        if n >= self.n_intersections:
            self.body_length = t - 1
            self.center_line = line
            self.body_line = [(h.index[-self.body_length], line), (h.index[-1], line)]
            return True
        return False
    
    def get_trend(self, h) -> bool:
        self.trend_line = [(), (h.index[-1], self.center_line)]
        tmin = -self.trend_maxsize + h.Low[-self.trend_maxsize:].argmin()
        tmax = -self.trend_maxsize + h.High[-self.trend_maxsize:].argmax()
        if h.High[tmax] - self.center_line > self.center_line - h.Low[tmin]:
            self.trend_line[0] = (h.index[tmax], h.High[tmax])
            self.trend_length = tmax
        else:
            self.trend_line[0] = (h.index[tmin], h.Low[tmin])    
            self.trend_length = tmin    
    
    def get_target(self, h):
        ptrg = h.Close.values[self.body_length]
        self.target = (ptrg - h.Close.values[0])/h.Close.values[0]
        self.target_line = [(h.index[0], self.center_line), (h.index[self.body_length], ptrg)]