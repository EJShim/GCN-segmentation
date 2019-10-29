from layers import *

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}

        for kwarg in kwargs.keys():            
            assert kwarg in allowed_kwargs, 'inValid keyword argument : ' + kwarg

        name = kwargs.get('name')

        if not name:
            name = self.__class__.__name__.lower()
        

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.output1 = None
        self.output2 = None
        self.output3 = None
        self.output1_2 = None
        self.output2_2 = None

        self.loss = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError
    
    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        #build sequential resenet model
        eltwise = [3,5,7,9,11,13,  19,21,23,25,27,29, 35,37,39,41,45]
        concat = [15, 31]
        self.activations.append(self.inputs)


        for idx, layer in enumerate(self.layers):
            hidden = layer(self.activations[-1]) ##Layer Should be implemented

            if idx in eltwise:
                hidden = tf.add(hidden, self.activations[-2]) * 0.5
            if idx in concat:
                hidden = tf.concat([hidden, self.activations[-2]], 1)
            self.activations.append(hidden)
        
        self.output1 = self.activations[15]
    

            



if __name__ == "__main__":
    print("ha")

    Model(name="d")
