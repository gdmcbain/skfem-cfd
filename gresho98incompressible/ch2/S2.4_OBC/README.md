# Open boundary conditions

§2.4.1 ‘Open boundary conditions’ of

* Gresho, P. M., Sani, R. L. (1998). *Incompressible Flow and the Finite Element Method. Volume One: Advection--Diffusion.* Chichester, West Sussex: Wiley

The boundary-value problem is ∂*T*/∂*t* = ∂²*T*/∂*x*² on 0 < _x_ < 1 subject to *T*(1) = 0 and the ‘open boundary condition’ *T*' (0) = −1.

The idea here is to integrate in time very accurately so that all error is due to spatial discretization and then see how the pure Neumann condition at the origin does.

![fig2.4-1](https://user-images.githubusercontent.com/1588947/140462915-783df196-7be4-47ce-b7c1-5058f5428312.png)

**Fig. 2.4-1** *Exact and approximation solutions at x = 0.*
