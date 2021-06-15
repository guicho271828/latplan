
"""
obsolete model classes for latplan AMA2
"""


from .model                  import *

# AMA2: state/action discriminator #############################################

class Discriminator(Network):
    """Base class for generic binary classifiers."""
    def _build_primary(self,input_shape):
        x = Input(shape=input_shape)
        N = input_shape[0] // 2

        y = Sequential([
            MyFlatten(),
            *[Sequential([BN(),
                          Dense(self.parameters["layer"],activation=self.parameters["activation"]),
                          Dropout(self.parameters["dropout"]),])
              for i in range(self.parameters["num_layers"]) ],
            Dense(1,activation="sigmoid")
        ])(x)

        self.loss = bce
        self.net = Model(x, y)

    def _report(self,test_both,**opts):
        from .util.np_distances import bce
        test_both(["BCE"], lambda data: bce(data, self.autoencode(data,**opts)))
        return
    def discriminate(self,data,**kwargs):
        self.load()
        return self.net.predict(data,**kwargs)
    def summary(self,verbose=False):
        self.net.summary()
        return self

class PUDiscriminator(Discriminator):
    """Subclass for PU-learning."""
    def _save(self,path=""):
        super()._save(path)
        print("saving PU discriminator C value")
        import os.path
        np.savez_compressed(self.local(os.path.join(path,f"c.npz")),c=K.get_value(self.c))

    def _load(self,path=""):
        super()._load(path)
        print("loading PU discriminator C value")
        import os.path
        try:
            with np.load(self.local(os.path.join(path,f"c.npz"))) as data:
                K.set_value(self.c, data["c"])
        except FileNotFoundError:
            print("failed to find weights for additional networks")

    def _build_around(self,input_shape):
        super()._build_around(input_shape)
        c = K.variable(0, name="c")
        self.c = c

        x = Input(shape=input_shape)
        s = self.net(x)
        y2 = wrap(s, s / c)
        self.pu = Model(x,y2)

    def discriminate(self,data,**kwargs):
        self.load()
        return self.pu.predict(data,**kwargs)

    def train(self,train_data,
              batch_size=1000,
              save=True,
              train_data_to=None,
              val_data=None,
              val_data_to=None,
              **kwargs):
        super().train(train_data,
                      batch_size=batch_size,
                      train_data_to=train_data_to,
                      val_data=val_data,
                      val_data_to=val_data_to,
                      save=False,
                      **kwargs)

        s = self.net.predict(val_data[val_data_to == 1],batch_size=batch_size)
        if np.count_nonzero(val_data_to == 1) > 0:
            c = s.mean()
            print("PU constant c =", c)
            K.set_value(self.c, c)
            # prevent saving before setting c
            if save:
                self.save()
        else:
            raise Exception("there are no positive data in the validation set; Training failed.")

class SimpleCAE(AE):
    """A Hack"""
    def build_encoder(self,input_shape):
        return [Reshape((*input_shape,1)),
                GaussianNoise(0.1),
                BN(),
                Convolution2D(self.parameters["clayer"],(3,3),
                              activation=self.parameters["activation"],padding="same", use_bias=False),
                Dropout(self.parameters["dropout"]),
                BN(),
                MaxPooling2D((2,2)),
                Convolution2D(self.parameters["clayer"],(3,3),
                              activation=self.parameters["activation"],padding="same", use_bias=False),
                Dropout(self.parameters["dropout"]),
                BN(),
                MaxPooling2D((2,2)),
                Convolution2D(self.parameters["clayer"],(3,3),
                              activation=self.parameters["activation"],padding="same", use_bias=False),
                Dropout(self.parameters["dropout"]),
                BN(),
                MaxPooling2D((2,2)),
                MyFlatten(),]

    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            Dense(self.parameters["layer"], activation="relu", use_bias=False),
            BN(),
            Dropout(self.parameters["dropout"]),
            Dense(self.parameters["layer"], activation="relu", use_bias=False),
            BN(),
            Dropout(self.parameters["dropout"]),
            Dense(data_dim, activation="sigmoid"),
            Reshape(input_shape),]

    def _build_primary(self,input_shape):
        _encoder = self.build_encoder(input_shape)
        _decoder = self.build_decoder(input_shape)

        x = Input(shape=input_shape)
        z = Sequential([MyFlatten(), *_encoder])(x)
        y = Sequential(_decoder)(MyFlatten()(z))

        z2 = Input(shape=K.int_shape(z)[1:])
        y2 = Sequential(_decoder)(MyFlatten()(z2))

        self.loss = bce
        self.encoder     = Model(x, z)
        self.decoder     = Model(z2, y2)
        self.net = Model(x, y)
        self.autoencoder = self.net


_combined = None
def combined_sd(states,sae,cae,sd3,**kwargs):
    global _combined
    if _combined is None:
        x = Input(shape=states.shape[1:])
        tmp = x
        if sd3.parameters["method"] == "direct":
            tmp = sd3.net(tmp)
        if sd3.parameters["method"] == "feature":
            tmp = sae.decoder(tmp)
            tmp = sae.features(tmp)
            tmp = sd3.net(tmp)
        if sd3.parameters["method"] == "cae":
            tmp = sae.decoder(tmp)
            tmp = cae.encoder(tmp)
            tmp = sd3.net(tmp)
        if sd3.parameters["method"] == "image":
            tmp = sae.decoder(tmp)
            tmp = cae.encoder(tmp)
            tmp = sd3.net(tmp)
        _combined = Model(x, tmp)
    return _combined.predict(states, **kwargs)



# AMA2: action autoencoder #####################################################

class ActionAE(AE):
    """A network which autoencodes the difference information.

    State transitions are not a 1-to-1 mapping in a sense that
    there are multiple applicable actions. So you cannot train a newtork that directly learns
    a transition S -> T .

    We also do not have action labels, so we need to cluster the actions in an unsupervised manner.

    This network trains a bidirectional mapping of (S,T) -> (S,A) -> (S,T), given that 
    a state transition is a function conditioned by the before-state s.

    It is not useful to learn a normal autoencoder (S,T) -> Z -> (S,T) because we cannot separate the
    condition and the action label.

    We again use gumbel-softmax for representing A."""
    def build_encoder(self,input_shape):
        return [
            *[
                Sequential([
                    Dense(self.parameters["aae_width"], activation=self.parameters["aae_activation"], use_bias=False),
                    BN(),
                    Dropout(self.parameters["dropout"]),])
                for i in range(self.parameters["aae_depth"]-1)
            ],
            Sequential([
                    Dense(self.parameters["N"]*self.parameters["M"]),
                    self.build_gs(),
            ]),
        ]

    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            *[
                Sequential([
                    Dense(self.parameters["aae_width"], activation=self.parameters["aae_activation"], use_bias=False),
                    BN(),
                    Dropout(self.parameters["dropout"]),])
                for i in range(self.parameters["aae_depth"]-1)
            ],
            Sequential([
                Dense(data_dim),
                self.build_bc(),
                Reshape(input_shape),]),]

    def _build_primary(self,input_shape):

        dim = np.prod(input_shape) // 2
        print("{} latent bits".format(dim))
        M, N = self.parameters["M"], self.parameters["N"]

        x = Input(shape=input_shape)

        pre = wrap(x,x[:,:dim],name="pre")
        suc = wrap(x,x[:,dim:],name="suc")

        print("encoder")
        _encoder = self.build_encoder([dim])
        action = ConditionalSequential(_encoder, pre, axis=1)(suc)

        print("decoder")
        _decoder = self.build_decoder([dim])
        suc_reconstruction = ConditionalSequential(_decoder, pre, axis=1)(MyFlatten()(action))
        y = Concatenate(axis=1)([pre,suc_reconstruction])

        action2 = Input(shape=(N,M))
        pre2    = Input(shape=(dim,))
        suc_reconstruction2 = ConditionalSequential(_decoder, pre2, axis=1)(MyFlatten()(action2))
        y2 = Concatenate(axis=1)([pre2,suc_reconstruction2])

        self.metrics.append(MAE)
        self.loss = BCE
        self.encoder     = Model(x, [pre,action])
        self.decoder     = Model([pre2,action2], y2)

        self.net = Model(x, y)
        self.autoencoder = self.net

    def encode_action(self,data,**kwargs):
        M, N = self.parameters["M"], self.parameters["N"]
        return self.encode(data,**kwargs)[1]

    def report(self,train_data,
               epoch=200,batch_size=1000,optimizer=Adam(0.001),
               test_data=None,
               train_data_to=None,
               test_data_to=None,):
        test_data     = train_data if test_data is None else test_data
        train_data_to = train_data if train_data_to is None else train_data_to
        test_data_to  = test_data  if test_data_to is None else test_data_to
        opts = {"verbose":0,"batch_size":batch_size}
        def test_both(msg, fn):
            print(msg.format(fn(train_data)))
            if test_data is not None:
                print((msg+" (validation)").format(fn(test_data)))
        self.autoencoder.compile(optimizer=optimizer, loss=bce)
        test_both("Reconstruction BCE: {}",
                  lambda data: self.autoencoder.evaluate(data,data,**opts))
        return self

    def plot(self,data,path,verbose=False,sae=None):
        self.load()
        dim = data.shape[1] // 2
        x = data
        _, z = self.encode(x) # _ == x
        y = self.decode([x[:,:dim],z])
        b = np.round(z)
        by = self.decode([x[:,:dim],b])

        x_pre, x_suc = squarify(x[:,:dim]), squarify(x[:,dim:])
        y_pre, y_suc = squarify(y[:,:dim]), squarify(y[:,dim:])
        by_pre, by_suc = squarify(by[:,:dim]), squarify(by[:,dim:])
        y_suc_r, by_suc_r = y_suc.round(), by_suc.round()

        if sae:
            x_pre_im, x_suc_im = sae.decode(x[:,:dim]), sae.decode(x[:,dim:])
            y_pre_im, y_suc_im = sae.decode(y[:,:dim]), sae.decode(y[:,dim:])
            by_pre_im, by_suc_im = sae.decode(by[:,:dim]), sae.decode(by[:,dim:])
            y_suc_r_im, by_suc_r_im = sae.decode(y[:,dim:].round()), sae.decode(by[:,dim:].round())
            _plot(self.local(path),
                  (x_pre_im, x_suc_im, squarify(np.squeeze(z)),
                   y_pre_im, y_suc_im, y_suc_r_im, squarify(np.squeeze(b)),
                   by_pre_im, by_suc_im, by_suc_r_im))
        else:
            _plot(self.local(path),
                  (x_pre, x_suc, squarify(np.squeeze(z)),
                   y_pre, y_suc, y_suc_r, squarify(np.squeeze(b)),
                   by_pre, by_suc, by_suc_r))
        return x,z,y,b,by

class CubeActionAE(ActionAE):
    """AAE with cube-like structure, developped for a compariason purpose."""
    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            *[
                Sequential([
                    Dense(self.parameters["aae_width"], activation=self.parameters["aae_activation"], use_bias=False),
                    BN(),
                    Dropout(self.parameters["dropout"]),])
                for i in range(self.parameters["aae_depth"]-1)
            ],
            Sequential([
                Dense(data_dim,use_bias=False),
                BN(),
            ]),
        ]

    def _build_primary(self,input_shape):

        dim = np.prod(input_shape) // 2
        print("{} latent bits".format(dim))
        M, N = self.parameters["M"], self.parameters["N"]

        x = Input(shape=input_shape)

        pre = wrap(x,x[:,:dim],name="pre")
        suc = wrap(x,x[:,dim:],name="suc")

        _encoder = self.build_encoder([dim])
        action = ConditionalSequential(_encoder, pre, axis=1)(suc)

        _decoder = self.build_decoder([dim])
        l_eff = Sequential(_decoder)(MyFlatten()(action))

        scaling = BN()
        l_pre = scaling(pre)

        l_suc = add([l_eff,l_pre])
        suc_reconstruction = self.build_bc()(l_suc)

        y = Concatenate(axis=1)([pre,suc_reconstruction])

        action2 = Input(shape=(N,M))
        pre2    = Input(shape=(dim,))
        l_pre2 = scaling(pre2)
        l_eff2 = Sequential(_decoder)(MyFlatten()(action2))
        l_suc2 = add([l_eff2,l_pre2])
        suc_reconstruction2 = self.build_bc()(l_suc2)
        y2 = Concatenate(axis=1)([pre2,suc_reconstruction2])

        self.metrics.append(MAE)
        self.loss = BCE
        self.encoder     = Model(x, [pre,action])
        self.decoder     = Model([pre2,action2], y2)

        self.net = Model(x, y)
        self.autoencoder = self.net


class ActionDiscriminator(Discriminator):
    def _build_primary(self,input_shape):
        A = 128
        N = input_shape[0] - A
        x = Input(shape=input_shape)
        pre    = wrap(x,tf.slice(x, [0,0], [-1,N]),name="pre")
        action = wrap(x,tf.slice(x, [0,N], [-1,A]),name="action")

        ys = []
        for i in range(A):
            _x = Input(shape=(N,))
            _y = Sequential([
                MyFlatten(),
                *[Sequential([BN(),
                              Dense(self.parameters["layer"],activation=self.parameters["activation"]),
                              Dropout(self.parameters["dropout"]),])
              for i in range(self.parameters["num_layers"]) ],
                Dense(1,activation="sigmoid")
            ])(_x)
            _m = Model(_x,_y,name="action_"+str(i))
            ys.append(_m(pre))

        ys = Concatenate()(ys)
        y  = Dot(-1)([ys,action])

        self.loss = bce
        self.net = Model(x, y)

class ActionPUDiscriminator(PUDiscriminator,ActionDiscriminator):
    pass

