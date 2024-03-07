import numpy as np
#PID控制器
class PIDController():
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_sum = 0.0
        self.last_error = 0.0
    def update(self, target, current, dt):
        error  = target-current
        self.error_sum += error * dt
        error_diff = (error - self.last_error)/dt
        #error_diff = (error - self.last_error)
        output = self.kp * error +self.ki * self.error_sum + self.kd * error_diff
        self.last_error = error
        return output

#鲁棒控制器
class AccelerateController():
    def __init__(self, tao1, tao2, Et,Ua=0.0):
        self.tao1 = tao1
        self.tao2 = tao2
        self.Et = Et
        self.Sv = 0
        self.Su = 0
        self.Ua = Ua

    def update_X(self,U_,dt,VL,PL):
        #delta_Ua = -self.Et*(self.Sv - VL) + self.Su
        #U = -1*self.tao2*(tanh(VL + self.tao1*PL)+tanh(VL)) - sat(delta_Ua,U_)
        U = -1*self.tao2*(self.tanh(VL + self.tao1*PL)+self.tanh(VL))
        acc = U
        #acc = U + self.Ua
        V_out = VL + acc*dt
        #更新Sv和Su
        Sv_1 = self.Sv - self.Et*(self.Sv - VL)*dt
        Su_1 = self.Su - self.Et*(self.Su + U)*dt
        self.Sv = Sv_1
        self.Su = Su_1
        return V_out
    
    def update_Y(self,U_,dt,VL,PL):
        delta_Ua = -1*self.Et*(self.Sv + VL) + self.Su
        U =-self.tao2*(self.tanh(VL + self.tao1*PL)+self.tanh(VL)) - self.sat(delta_Ua,U_)
        acc = U
        V_out = VL + acc*dt
        #更新Sv和Su
        Sv_1 = self.Sv - self.Et*(self.Sv + VL)*dt
        Su_1 = self.Su - self.Et*(self.Su + U)*dt
        self.Sv = Sv_1
        self.Su = Su_1
        return V_out
    
    def update_Z(self,U_,dt,VL,PL):
        delta_Ua = -1*self.Et*(self.Sv + VL) + self.Su
        U = 9.8 - self.tao2*(self.tanh(VL + self.tao1*PL)+self.tanh(VL)) - self.sat(delta_Ua,U_)
        acc = U - 9.8
        V_out = VL + acc*dt
        #更新Sv和Su
        Sv_1 = self.Sv - self.Et*(self.Sv + VL)*dt
        Su_1 = self.Su - self.Et*(self.Su + U - 9.8)*dt
        self.Sv = Sv_1
        self.Su = Su_1
        return V_out
    
    # 饱和函数（saturation function）
    def sat(self,x,u):
        if x < -u:
            return -u
        elif x > u:
            return u
        else:
            return x 

    # 双曲正切函数（tanh function）
    def tanh(self,x):
        return np.tanh(x)