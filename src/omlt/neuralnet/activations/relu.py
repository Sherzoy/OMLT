import pyomo.environ as pyo
import pyomo.mpec as mpec


def bigm_relu_activation_constraint(net_block, net, layer_block, layer):
    r"""
    Big-M ReLU activation formulation.

    Generates the constraints for the ReLU activation function.

    .. math::

        \begin{align*}
        z_i &= \text{max}(0, \hat{z_i}) && \forall i \in N
        \end{align*}

    The Big-M formulation for the i-th node is given by:

    .. math::

        \begin{align*}
        z_i &\geq \hat{z_i} \\
        z_i &\leq \hat{z_i} - l(1-\sigma) \\
        z_i &\leq u(\sigma) \\
        \sigma &\in \{0, 1\}
        \end{align*}

    where :math:`l` and :math:`u` are, respectively, lower and upper bounds of :math:`\hat{z_i}`.
    """
    layer_block.q_relu = pyo.Var(layer.output_indexes, within=pyo.Binary)

    layer_block._z_lower_bound_relu = pyo.Constraint(layer.output_indexes)
    layer_block._z_lower_bound_zhat_relu = pyo.Constraint(layer.output_indexes)
    layer_block._z_upper_bound_relu = pyo.Constraint(layer.output_indexes)
    layer_block._z_upper_bound_zhat_relu = pyo.Constraint(layer.output_indexes)

    # set dummy parameters here to avoid warning message from Pyomo
    layer_block._big_m_lb_relu = pyo.Param(
        layer.output_indexes, default=-1e6, mutable=True
    )
    layer_block._big_m_ub_relu = pyo.Param(
        layer.output_indexes, default=1e6, mutable=True
    )

    for output_index in layer.output_indexes:
        lb, ub = layer_block.zhat[output_index].bounds
        layer_block._big_m_lb_relu[output_index] = lb
        layer_block.z[output_index].setlb(0)

        layer_block._big_m_ub_relu[output_index] = ub
        layer_block.z[output_index].setub(max(0, ub))

        layer_block._z_lower_bound_relu[output_index] = layer_block.z[output_index] >= 0

        layer_block._z_lower_bound_zhat_relu[output_index] = (
            layer_block.z[output_index] >= layer_block.zhat[output_index]
        )

        layer_block._z_upper_bound_relu[output_index] = (
            layer_block.z[output_index]
            <= layer_block._big_m_ub_relu[output_index]
            * layer_block.q_relu[output_index]
        )

        layer_block._z_upper_bound_zhat_relu[output_index] = layer_block.z[
            output_index
        ] <= layer_block.zhat[output_index] - layer_block._big_m_lb_relu[
            output_index
        ] * (
            1.0 - layer_block.q_relu[output_index]
        )

def bigm_leaky_relu_activation_constraint(net_block, net, layer_block, layer):
    # instantiate
    layer_block.qb=pyo.Var(layer.output_indexes,within=pyo.Binary)
    layer_block.c1=pyo.Constraint(layer.output_indexes)
    layer_block.c2=pyo.Constraint(layer.output_indexes)
    layer_block.c3=pyo.Constraint(layer.output_indexes)
    layer_block.c4=pyo.Constraint(layer.output_indexes)
    
    # Extract the value of alpha
    a = layer.alpha

    # integrate
    for output_index in layer.output_indexes:
        lb,ub = layer_block.zhat[output_index].bounds
        assert(lb<=ub)
        layer_block.z[output_index].setlb(max(a*lb,lb))
        layer_block.z[output_index].setub(max(a*ub,ub))
        q=layer_block.qb[output_index]
        zhat=layer_block.zhat[output_index]
        z=layer_block.z[output_index]
        layer_block.c1[output_index]=z>=a*zhat
        layer_block.c2[output_index]=z>=zhat
        layer_block.c3[output_index]=z<=zhat-(1-a)*lb*(1-q)
        layer_block.c4[output_index]=z<=a*zhat+(1-a)*ub*q

def bigm_para_relu_activation_constraint(net_block, net, layer_block, layer):
    #instantiate
    layer_block.qb=pyo.Var(layer.output_indexes,within=pyo.Binary)
    layer_block.c1=pyo.Constraint(layer.output_indexes)
    layer_block.c2=pyo.Constraint(layer.output_indexes)
    layer_block.c3=pyo.Constraint(layer.output_indexes)
    layer_block.c4=pyo.Constraint(layer.output_indexes)

    alph = layer.alpha  # Extract the value of alpha
    count = 0
    for output_index in layer.output_indexes:
        a = alph[count]     
        if(a>=0):
            lb,ub = layer_block.zhat[output_index].bounds
            assert(lb<=ub)
            layer_block.z[output_index].setlb(a*lb)
            layer_block.z[output_index].setub(ub)
        else:
            lb,ub = layer_block.zhat[output_index].bounds
            assert(lb<=ub)
            layer_block.z[output_index].setlb(0)
            layer_block.z[output_index].setub(ub)
        
        if(a>=1):
            q=layer_block.qb[output_index]
            zhat=layer_block.zhat[output_index]
            z=layer_block.z[output_index]
            layer_block.c1[output_index]=z>=a*zhat+(1-a)*ub*q
            layer_block.c2[output_index]=z>=zhat-lb*(1-a)*(1-q)
            layer_block.c3[output_index]=z<=zhat
            layer_block.c4[output_index]=z<=a*zhat

        else:
            q=layer_block.qb[output_index]
            zhat=layer_block.zhat[output_index]
            z=layer_block.z[output_index]
            layer_block.c1[output_index]=z<=a*zhat+(1-a)*ub*q
            layer_block.c2[output_index]=z<=zhat-lb*(1-a)*(1-q)
            layer_block.c3[output_index]=z>=zhat
            layer_block.c4[output_index]=z>=a*zhat

        count+=1
        

class ComplementarityReLUActivation:
    r"""
    Complementarity-based ReLU activation forumlation.

    Generates the constraints for the ReLU activation function.

    .. math::

        \begin{align*}
        z_i &= \text{max}(0, \hat{z_i}) && \forall i \in N
        \end{align*}

    The complementarity-based formulation for the i-th node is given by:

    .. math::

        \begin{align*}
        0 &\leq z_i \perp (z-\hat{z_i}) \geq 0
        \end{align*}

    """

    def __init__(self, transform=None):
        if transform is None:
            transform = "mpec.simple_nonlinear"
        self.transform = transform

    def __call__(self, net_block, net, layer_block, layer):
        layer_block._complementarity = mpec.Complementarity(
            layer.output_indexes, rule=_relu_complementarity
        )
        xfrm = pyo.TransformationFactory(self.transform)
        xfrm.apply_to(layer_block)


def _relu_complementarity(b, *output_index):
    return mpec.complements(
        b.z[output_index] - b.zhat[output_index] >= 0, b.z[output_index] >= 0
    )
