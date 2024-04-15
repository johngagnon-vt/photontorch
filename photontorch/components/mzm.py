""" MZIs are 4-port components coupling two waveguides together. """

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np
from scipy.interpolate import interp1d
import os

## Relative
from .component import Component
from ..nn.nn import Parameter, Buffer


#########################
## Directional Coupler ##
#########################


class Mzm(Component):
    """An MZm is a component with 2 ports.

    

    Terms::

                    _[2*theta]_
          _______  /           \  ___
      0__/       \/             \/    \____1
         \_______/\_____________/\___/

    Note:
        This MZM implementation assumes the armlength difference is too small to have
        a noticable delay difference between the arms, i.e. only the phase difference matters

    """

    num_ports = 2

    def __init__(
        self,
        phi=0,
        theta=0,
        neff=2.34,
        ng=3.40,
        wl0=1.55e-6,
        length=1e-5,
        loss=0,
        trainable=True,
        normalize = True,
        debug_print = False,
        print_phase=False,
        name=None,
        S=None,
    ):
        """
        Args:
            phi (float): input phase
            theta (float): phase difference between the arms
            neff (float): effective index of the waveguide
            ng (float): group index of the waveguide
            wl0 (float): the center wavelength for which neff is defined.
            length (float): length of the waveguide in meter.
            loss (float): loss in the waveguide [dB/m]
            trainable (bool): whether phi and theta are trainable
            name (optional, str): name of this specific MZI
        """
        super(Mzm, self).__init__(name=name)

        parameter = Parameter if trainable else Buffer
        self.print_phase = print_phase
        self.ng = float(ng)
        self.neff = float(neff)
        self.length = float(length)
        self.loss = float(loss)
        self.wl0 = float(wl0)
        self.normalize = normalize
        self.debug_print = debug_print
        self.phi = parameter(torch.tensor(phi, dtype=torch.float64, device=self.device))
        self.theta = parameter(
            torch.tensor(theta, dtype=torch.float64, device=self.device)
        )
        if S != None:
            self.S = S

    def set_delays(self, delays):
        delays[:] = self.ng * self.length / self.env.c
        
    def get_voltage_index(self,target_value):
        
        
        
        
        
        current_directory = os.path.dirname(os.path.abspath(__file__))
        lookup_tables_directory = os.path.join(current_directory, 'lookup_tables')
        file_path = os.path.join(lookup_tables_directory, 'Heater_Voltage_to_theta.txt')
        values = np.loadtxt(file_path)
        target_value = (target_value+values[0])
        #(target_value-values[0])%(2*np.pi)+values[0]#putting  the value from min to min+2pi
        closest_index = np.abs(values - target_value).argmin()
        return closest_index
    
    
    
    
    def get_value_at_index(self,file_name, index):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        lookup_tables_directory = os.path.join(current_directory, 'lookup_tables')
        file_path = os.path.join(lookup_tables_directory, file_name)
        values = np.loadtxt(file_path)
        if 0 <= index < len(values):
            return values[index] 
        else:
            return None
        
   
    
    def normalize_loss(self,out1,out2,phi):
        
        
        phi1 = np.arctan2(out1[1], out1[0])
        phi2 = np.arctan2(out2[1], out2[0])
        
        
        cos_phi1 = np.cos(phi1)
        cos_phi2 = np.cos(phi2)
        
        
        #this 100/pi is a scaling factor due to the output of the Ansys simulation being in different units
        #may need to be changed if using a different simulation software
        cos_theta1 = 100*out1[0]/cos_phi1/np.pi 
        sin_theta2 = -100*out2[0]/cos_phi2/np.pi
        
        if( self.normalize == True):
 
            theta = np.arctan2(sin_theta2,cos_theta1)
            cos_theta1=np.cos(theta)
            sin_theta2=np.sin(theta)
            
            sin_phi2=np.sin(phi2)
            
            phi1=np.arctan2(sin_phi2,cos_phi1)

        
        
        fixed_out1 = [np.sign(out1[0])*np.abs(np.cos(phi1+phi) * cos_theta1),
                      np.sign(out1[1])*np.abs(np.sin(phi1+phi) * cos_theta1)]#doing this just to ensure signs, as the sin,cos may be outside of its quadrants
        
        fixed_out2 = [np.sign(out2[0])*np.abs(np.cos(phi2+phi) * sin_theta2),
                      np.sign(out2[1])*np.abs(np.sin(phi2+phi) * sin_theta2)]#doing this just to ensure signs, as the sin,cos may be outside of its quadrants
        
    
        if (self.debug_print == True):
            print("-----------------------------------------")
            
            sin_phi1 = np.sin(phi1)
            sin_phi2 = np.sin(phi2)
            
            
            cos_theta1b = 100*out1[1]/sin_phi1/np.pi
            sin_theta2b = -100*out2[1]/sin_phi2/np.pi
            
            theta = np.arctan2(sin_theta2,cos_theta1)
            theta1 = np.arccos(cos_theta1)
            theta2 = np.arcsin(sin_theta2)
            print("output 1: ", out1)
            print("output 2: ", out2)
            if (np.abs(cos_theta1 - cos_theta1b)>.001*np.abs(cos_theta1)):
                print("error on cos",cos_theta1," != ",cos_theta1b )
                
            if (np.abs(sin_theta2 - sin_theta2b)>.001*np.abs(sin_theta2)):
                print("error on sin",sin_theta2," != ",sin_theta2b )
            
            print("phi1: ",phi1,"phi2: ",phi2)
            print("cos phi1: ",cos_phi1,"sin phi1: ",sin_phi1)
            print("cos phi2: ",cos_phi2,"sin phi2: ",sin_phi2)
            print("cos theta 1: ", cos_theta1,"sin theta 2: ", sin_theta2)
            print("magnitude: " , np.sqrt(np.square(cos_theta1)+np.square(sin_theta2)))
            print("theta 1: ", theta1,"theta 2: ", theta2)
            print("theta: ",theta)
            print("fixed output 1: ", fixed_out1)
            print("fixed output 2: ", fixed_out2)
            print("-----------------------------------------")
        
        
        return fixed_out1,fixed_out2    
        
        
            
    
    def set_S(self, S):
        
        A = np.zeros((2, 4, 4))
        theta = float(self.theta)
        index = self.get_voltage_index(theta)
        in1_out1,in1_out2=self.normalize_loss([self.get_value_at_index('in1_out1_re.txt',index),self.get_value_at_index('in1_out1_im.txt',index)],
                                              [self.get_value_at_index('in1_out2_re.txt',index),self.get_value_at_index('in1_out2_im.txt',index)],
                                              0)
        A[0, 2, 3] = in1_out1[0]
        A[0, 1, 3] = in1_out2[0]
        A[1, 1, 3] = in1_out2[1]
        A[1, 2, 3] = in1_out1[1]
        
        in2_out1,in2_out2=self.normalize_loss([self.get_value_at_index('in2_out1_re.txt',index),self.get_value_at_index('in2_out1_im.txt',index)],
                                              [self.get_value_at_index('in2_out2_re.txt',index),self.get_value_at_index('in2_out2_im.txt',index)],
                                              0)
        A[0, 0, 2] = in2_out1[0]
        A[0, 0, 1] = in2_out2[0]
        A[1, 0, 1] = in2_out2[1]
        A[1, 0, 2] = in2_out1[1]


        #print("A",A)

        out_real = np.sum(A[0,:,:])/2
        out_im =np.sum(A[1,:,:])/2
        
        S[0,:,0,1] = S[0,:,1,0] = out_real
        S[1,:,0,1] = S[1,:,1,0] = out_im
        if self.print_phase:
            print(np.arctan2(out_im,out_real))
        torch.Size([2, 1, 2, 2])
        
        
        
        return S# * 10 ** (-loss / 20)  # 20 bc loss is defined on power.


        
        
    def action(self, t, x_in, x_out):
        """Nonlinear action of the component on its active nodes

        Args:
            t (float): the current time in the simulation
            x_in (torch.Tensor[#active nodes, 2, #wavelengths, #batches]): the input tensor
                used to define the action
            x_out (torch.Tensor[#active nodes, 2, #wavelengths, #batches]): the output
                tensor. The result of the action should be stored in the
                elements of this tensor.

        """
