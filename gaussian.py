from model import AE, GumbelAE, GumbelAE2, ConvolutionalGumbelAE, default_networks

class GaussianSample:
    count = 0
    
    def __init__(self,G):
        self.G = G
        
    def call(self,args):
        mean, log_var = args
        epsilon = K.random_normal(shape=K.shape(mean), mean=0., std=1.0)
        return mean + K.exp(log_var / 2) * epsilon
    
    def __call__(self,prev):
        GaussianSample.count += 1
        c = GaussianSample.count-1
        prev = flatten(prev)
        mean    = Dense(self.G,name="gmean_{}".format(c))(prev)
        log_var = Dense(self.G,name="glogvar_{}".format(c))(prev)
        self.mean, self.log_var = mean, log_var
        return Lambda(self.call,name="gaussian_{}".format(c))([mean,log_var])
    
    def loss(self):
        return - 0.5 * K.mean(
            1 + self.log_var - K.square(self.mean) - K.exp(self.log_var),
            axis=-1)

class GaussianAE(AE):
    def __init__(self,path,parameters={}):
        if 'G' not in parameters:
            parameters['G'] = 10
        super().__init__(path,parameters)
        
    def _build(self,input_shape):
        data_dim = np.prod(input_shape)
        print("input_shape:{}, flattened into {}".format(input_shape,data_dim))
        G = self.parameters['G']
        x = Input(shape=input_shape)
        x_flat = flatten(x)
        pre_encoded = Sequential(self.build_encoder(input_shape))(x_flat)
        gauss  = GaussianSample(G)
        z = gauss (pre_encoded)
        _decoder = self.build_decoder(input_shape)
        y  = Sequential(_decoder)(z)

        z2 = Input(shape=(G,))
        y2 = Sequential(_decoder)(z2)
        
        def loss(x, y):
            kl_loss = gauss.loss()
            reconstruction_loss = bce(K.reshape(x,(K.shape(x)[0],data_dim,)),
                                      K.reshape(y,(K.shape(x)[0],data_dim,)))
            return reconstruction_loss + kl_loss

        self.loss = loss
        self.net         = Model(x, y)
        self.encoder     = Model(x, z)
        self.decoder     = Model(z2,y2)
        self.autoencoder = self.net

    def plot(self,data,path):
        self.load()
        xs = data
        zs = self.encode(xs)
        ys = self.decode(zs)
        G = self.parameters['G']
        import math
        root = math.sqrt(G)
        l1 = math.floor(root)
        if l1*l1 == G:
            _zs = zs.reshape((-1,l1,l1))
        else:
            l2 = math.ceil(root)
            size = l1*l2
            _zs = np.concatenate((zs,np.ones((zs.shape[0],size-G))),axis=1).reshape((-1,l1,l2))
        images = []
        from plot import plot_grid
        for seq in zip(xs, _zs, ys):
            images.extend(seq)
        plot_grid(images, path=self.local(path))
        return xs,zs,ys
        
class GaussianGumbelAE(GumbelAE):
    def __init__(self,path,parameters={}):
        if 'G' not in parameters:
            parameters['G'] = 10
        super().__init__(path,parameters)

    def _build(self,input_shape):
        data_dim = np.prod(input_shape)
        print("input_shape:{}, flattened into {}".format(input_shape,data_dim))
        M, N, G = self.parameters['M'], self.parameters['N'], self.parameters['G'], 
        x = Input(shape=input_shape)
        x_flat = flatten(x)
        pre_encoded = Sequential(self.build_encoder(input_shape))(x_flat)
        gumbel = GumbelSoftmax(N,M,
                               self.min_temperature,
                               self.max_temperature,
                               self.anneal_rate)
        gauss  = GaussianSample(G)
        z_cat   = gumbel(pre_encoded)
        z_gauss = gauss (pre_encoded)
        z_cat_flat = flatten(z_cat)
        z = merge([z_cat_flat, z_gauss], mode='concat')
        _decoder = self.build_decoder(input_shape)
        y  = Sequential(_decoder)(z)

        z_gzero = Input(shape=(G,))
        z2_cat  = Input(shape=(N,M))
        z2_cat_flat = flatten(z2_cat)
        z2 = Lambda(K.concatenate)([z2_cat_flat, z_gzero])
        y2 = Sequential(_decoder)(z2)
        
        def loss(x, y):
            kl_loss = gumbel.loss() + gauss.loss()
            reconstruction_loss = bce(K.reshape(x,(K.shape(x)[0],data_dim,)),
                                      K.reshape(y,(K.shape(x)[0],data_dim,)))
            return reconstruction_loss + kl_loss

        self.callbacks.append(LambdaCallback(on_epoch_end=gumbel.cool))
        self.custom_log_functions['tau'] = lambda: K.get_value(gumbel.tau)
        self.loss = loss
        self.net         = Model(x, y)
        self.encoder     = Model(x, z_cat)
        self.decoder     = Model([z2_cat,z_gzero],y2)
        self.autoencoder = self.net
        
    def decode(self,data,**kwargs):
        self.load()
        return self.decoder.predict([
            data,
            np.zeros((data.shape[0],self.parameters['G']))],**kwargs)

class GaussianGumbelAE2(GumbelAE2,GaussianGumbelAE):
    def _build(self,input_shape):
        data_dim = np.prod(input_shape)
        print("input_shape:{}, flattened into {}".format(input_shape,data_dim))
        M, N, G = self.parameters['M'], self.parameters['N'], self.parameters['G'], 
        x = Input(shape=input_shape)
        x_flat = flatten(x)
        pre_encoded = Sequential(self.build_encoder(input_shape))(x_flat)

        gumbel1 = GumbelSoftmax(N,M,        self.min_temperature, self.max_temperature, self.anneal_rate)
        gumbel2 = GumbelSoftmax(data_dim,2, self.min_temperature, self.max_temperature, self.anneal_rate)
        gumbel3 = GumbelSoftmax(data_dim,2, self.min_temperature, self.max_temperature, self.anneal_rate)
        gauss   = GaussianSample(G)
        _decoder = self.build_decoder(input_shape)
        
        z_cat      = gumbel1(pre_encoded)
        z_cat_flat = flatten(z_cat)
        z_gauss    = gauss  (pre_encoded)
        z_flat     = merge([z_cat_flat, z_gauss], mode='concat')

        y = Sequential([
            *_decoder,
            gumbel2,
            Lambda(take_true),
            Reshape(input_shape)
        ])(z_flat)
            
        z2_gzero = Input(shape=(G,))
        z2_cat   = Input(shape=(N,M))
        z2_cat_flat = flatten(z2_cat)
        # z2_flat = merge([z2_cat_flat, z_gzero], mode='concat')
        z2_flat = Lambda(K.concatenate)([z2_cat_flat, z2_gzero])
        y2 = Sequential([
            *_decoder,
            gumbel3,
            Lambda(take_true),
            Reshape(input_shape)
        ])(z2_flat)

        def loss(x, y):
            kl_loss = gumbel1.loss() + gumbel2.loss() + gauss.loss()
            reconstruction_loss = bce(K.reshape(x,(K.shape(x)[0],data_dim,)),
                                      K.reshape(y,(K.shape(x)[0],data_dim,)))
            return reconstruction_loss + kl_loss

        self.callbacks.append(LambdaCallback(on_epoch_end=gumbel1.cool))
        self.callbacks.append(LambdaCallback(on_epoch_end=gumbel2.cool))
        self.callbacks.append(LambdaCallback(on_epoch_end=gumbel3.cool))
        self.custom_log_functions['tau'] = lambda: K.get_value(gumbel1.tau)
        self.loss = loss
        self.net         = Model(x, y)
        self.encoder     = Model(x, z_cat)
        self.decoder     = Model([z2_cat,z2_gzero],y2)
        self.autoencoder = self.net

class GaussianConvolutionalGumbelAE(GaussianGumbelAE,ConvolutionalGumbelAE):
    pass

class GaussianConvolutionalGumbelAE2(GaussianGumbelAE2,ConvolutionalGumbelAE):
    pass




default_networks = {'gauss':GaussianAE,
                    'fcg':GaussianGumbelAE,'convg':GaussianConvolutionalGumbelAE,
                    'fcg2':GaussianGumbelAE2, 'convg2': GaussianConvolutionalGumbelAE2,
                    *default_networks}

