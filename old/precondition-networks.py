
# other action mapping network variations

class NonDetActionMixin:
    "Non-Deterministic mapping from the current state to an action"
    def build_action_encoder(self):
        return [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"])],
            Sequential([
                        Dense(self.parameters['num_actions']),
                        self.build_gs(N=1,
                                      M=self.parameters['num_actions'],
                                      offset=self.parameters["aae_delay"],
                                      beta=1.0),
            ]),
        ]
    def _action(self,z_pre,z_suc):
        self.action_encoder_net = self.build_action_encoder()

        N = self.parameters['N']
        transition = Input(shape=(N*2,))
        pre2 = wrap(transition, transition[:,:N])
        self.action = Model(transition, Sequential(self.action_encoder_net)(pre2))

        return Sequential(self.action_encoder_net)(z_pre)



# idea that has not been implemented/tested AFAIR
# kept here just as a future reminder
class NoSucActionMixin:
    pass


class DiffDetActionMixin(DetActionMixin):
    "Deterministic mapping from a state difference to an action"
    def _action(self,z_pre,z_suc):
        self.action_encoder_net = self.build_action_encoder()

        N = self.parameters['N']
        transition = Input(shape=(N*2,))
        pre2 = wrap(transition, transition[:,:N])
        suc2 = wrap(transition, transition[:,N:])
        diff2 = wrap(pre2, pre2-suc2)
        self.action = Model(transition, Sequential(self.action_encoder_net)(diff2))

        z_diff = wrap(z_pre, z_pre-z_suc)
        return Sequential(self.action_encoder_net)(z_diff)


class DetPreActionMixin(DetActionMixin):
    """Deterministic mapping from a state pair to an action, but consists of two parts.

The first part is a precondition subnetwork that only takes the current state and predicts the applicability for all actions.
The output size is (1, action).

The second part takes both the current and the successor states and predicts an action.
The output size is (1, action).

The applicability is always penalized so that it propose less candidates.
Two networks are also penalized when the predicted action is not predicted as applicable
(which increases the applicability vector, and decreases the action vector.)

"""
    def build_action_logit_encoder(self):
        A=self.parameters['num_actions']
        return [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"])],
            Sequential([
                Dense(A,use_bias=False),
                BN(),
                Reshape((1,A)),
            ]),
        ]
    def build_applicability_logit_encoder(self):
        # this precondition network does not necessarily encode conjunctions.
        A=self.parameters['num_actions']
        return [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"])],
            Sequential([
                Dense(A,use_bias=False),
                BN(),
                Reshape((1,A)),
            ]),
        ]
    def _action(self,z_pre,z_suc):

        N=self.parameters['N']
        A=self.parameters['num_actions']

        self.act_encoder_net = self.build_action_logit_encoder()
        self.app_encoder_net = self.build_applicability_logit_encoder()

        app_activation = rounded_sigmoid()
        act_activation = self.build_gs(N=1,
                                       M=A,
                                       offset=self.parameters["aae_delay"],
                                       alpha=-1.0)

        l_app        = Sequential(self.app_encoder_net)(z_pre)
        l_act        = ConditionalSequential(self.act_encoder_net, z_pre, axis=1)(z_suc)
        app          = app_activation(l_app)
        act          = act_activation(l_act)

        if self.parameters["filter_mode"] == "mul":
            filtered_act = multiply([app, act])
        elif self.parameters["filter_mode"] == "min":
            filtered_act = minimum([app, act])
        elif self.parameters["filter_mode"] == "smin":
            filtered_act = wrap(act,smooth_min(app, act))

        alpha = StepSchedule(schedule={
            0:0,
            (self.parameters["epoch"]*self.parameters["conjunctive_delay"]):self.parameters["conjunctive"],
        })
        self.callbacks.append(LambdaCallback(on_epoch_end=alpha.update))
        beta = StepSchedule(schedule={
            0:0,
            (self.parameters["epoch"]*self.parameters["applicability_delay"]):self.parameters["applicability"],
        })
        self.callbacks.append(LambdaCallback(on_epoch_end=beta.update))

        self.net.add_loss(
            # conjunctive loss : decrease app as much as possible
            K.in_train_phase(alpha.variable * K.mean(app),0.0))
        self.net.add_loss(
            # applicability loss : l_app must be larger than l_act
            K.in_train_phase(beta.variable  * K.mean(K.maximum(act - app, 0.0)),0.0))

        def applicability(x,y):
            return K.mean(K.maximum(act - app, 0))
        def conjunctive(x,y):
            return K.mean(app)
        self.metrics.append(applicability)
        self.metrics.append(conjunctive)

        transition = Input(shape=(N*2,))
        pre2 = wrap(transition, transition[:,:N])
        suc2 = wrap(transition, transition[:,N:])
        l_act2 = ConditionalSequential(self.act_encoder_net, pre2, axis=1)(suc2)
        act2 = act_activation(l_act2)
        self.action = Model(transition, act2)

        pre3 = Input(shape=(N,))
        l_app2 = Sequential(self.app_encoder_net)(pre3)
        app2 = app_activation(l_app2)
        self.applicable = Model(pre3, app2)

        return filtered_act


class DetRelaxedConjActionMixin(DetPreActionMixin):
    def build_applicability_logit_encoder(self):
        A=self.parameters['num_actions']
        return [
            # limited to 1 layer.
            # *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"])],
            Sequential([
                BN(),
                Dense(A,use_bias=False),
                BN(),
                Reshape((1,A)),
            ]),
        ]


class DetLogitConjActionMixin(DetRelaxedConjActionMixin):
    def _action(self,z_pre,z_suc):

        N=self.parameters['N']
        A=self.parameters['num_actions']

        self.act_encoder_net = self.build_action_logit_encoder()
        self.app_encoder_net = self.build_applicability_logit_encoder()

        app_activation = rounded_sigmoid()
        act_activation = self.build_gs(N=1,
                                       M=A,
                                       offset=self.parameters["aae_delay"],
                                       alpha=-1.0)

        l_app        = Sequential(self.app_encoder_net)(z_pre)
        l_act        = ConditionalSequential(self.act_encoder_net, z_pre, axis=1)(z_suc)
        app          = app_activation(l_app)
        act          = act_activation(l_act)

        if self.parameters["filter_mode"] == "mul":
            filtered_act = multiply([app, act])
        elif self.parameters["filter_mode"] == "min":
            filtered_act = minimum([app, act])
        elif self.parameters["filter_mode"] == "smin":
            filtered_act = wrap(act,smooth_min(app, act))

        alpha = StepSchedule(schedule={
            0:0,
            (self.parameters["epoch"]*self.parameters["conjunctive_delay"]):self.parameters["conjunctive"],
        })
        self.callbacks.append(LambdaCallback(on_epoch_end=alpha.update))
        beta = StepSchedule(schedule={
            0:0,
            (self.parameters["epoch"]*self.parameters["applicability_delay"]):self.parameters["applicability"],
        })
        self.callbacks.append(LambdaCallback(on_epoch_end=beta.update))

        self.net.add_loss(
            # conjunctive loss : decrease app as much as possible
            K.in_train_phase(alpha.variable * K.mean(l_app),0.0))
        self.net.add_loss(
            # applicability loss : l_app must be larger than l_act
            K.in_train_phase(beta.variable  * K.mean(K.maximum(act - app, 0.0)),0.0))

        def applicability(x,y):
            return K.mean(K.maximum(act - app, 0))
        def conjunctive(x,y):
            return K.mean(app)
        self.metrics.append(applicability)
        self.metrics.append(conjunctive)

        transition = Input(shape=(N*2,))
        pre2 = wrap(transition, transition[:,:N])
        suc2 = wrap(transition, transition[:,N:])
        l_act2 = ConditionalSequential(self.act_encoder_net, pre2, axis=1)(suc2)
        act2 = act_activation(l_act2)
        self.action = Model(transition, act2)

        pre3 = Input(shape=(N,))
        l_app2 = Sequential(self.app_encoder_net)(pre3)
        app2 = app_activation(l_app2)
        self.applicable = Model(pre3, app2)

        return filtered_act


class ConcreteDiffDetConditionalEffectTransitionAE(HammingMixin, ZeroSuppressMixin, DiffDetActionMixin, ConditionalEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass


class ConcreteNonDetConditionalEffectTransitionAE (HammingMixin, ZeroSuppressMixin, NonDetActionMixin, ConditionalEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass



class ConcreteDiffDetBoolMinMaxEffectTransitionAE(HammingMixin, ZeroSuppressMixin, DiffDetActionMixin, BoolMinMaxEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass

class ConcreteNonDetBoolMinMaxEffectTransitionAE (HammingMixin, ZeroSuppressMixin, NonDetActionMixin, BoolMinMaxEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass


class ConcreteDiffDetBoolSmoothMinMaxEffectTransitionAE(HammingMixin, ZeroSuppressMixin, DiffDetActionMixin, BoolSmoothMinMaxEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass

class ConcreteNonDetBoolSmoothMinMaxEffectTransitionAE (HammingMixin, ZeroSuppressMixin, NonDetActionMixin, BoolSmoothMinMaxEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass


class ConcreteDiffDetBoolAddEffectTransitionAE(HammingMixin, ZeroSuppressMixin, DiffDetActionMixin, BoolAddEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass

class ConcreteNonDetBoolAddEffectTransitionAE (HammingMixin, ZeroSuppressMixin, NonDetActionMixin, BoolAddEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass


class ConcreteDiffDetLogitAddEffectTransitionAE(HammingMixin, ZeroSuppressMixin, DiffDetActionMixin, LogitAddEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass

class ConcreteNonDetLogitAddEffectTransitionAE (HammingMixin, ZeroSuppressMixin, NonDetActionMixin, LogitAddEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass


class ConcreteDiffDetLogitAddEffect2TransitionAE(HammingMixin, ZeroSuppressMixin, DiffDetActionMixin, LogitAddEffect2Mixin, ConcreteLatentMixin, TransitionAE):
    pass

class ConcreteNonDetLogitAddEffect2TransitionAE (HammingMixin, ZeroSuppressMixin, NonDetActionMixin, LogitAddEffect2Mixin, ConcreteLatentMixin, TransitionAE):
    pass


class ConcreteDiffDetNormalizedLogitAddEffectTransitionAE(HammingMixin, ZeroSuppressMixin, DiffDetActionMixin, NormalizedLogitAddEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass

class ConcreteNonDetNormalizedLogitAddEffectTransitionAE (HammingMixin, ZeroSuppressMixin, NonDetActionMixin, NormalizedLogitAddEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass


class ConcreteDetPreNormalizedLogitAddEffectTransitionAE    (HammingMixin, ZeroSuppressMixin, DetPreActionMixin, NormalizedLogitAddEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass

class ConcreteDetRelaxedConjNormalizedLogitAddEffectTransitionAE    (HammingMixin, ZeroSuppressMixin, DetRelaxedConjActionMixin, NormalizedLogitAddEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass

class ConcreteDetLogitConjNormalizedLogitAddEffectTransitionAE    (HammingMixin, ZeroSuppressMixin, DetLogitConjActionMixin, NormalizedLogitAddEffectMixin, ConcreteLatentMixin, TransitionAE):
    pass

# class QuantizedDetConditionalEffectTransitionAE    (HammingMixin, ZeroSuppressMixin, DetActionMixin, ConditionalEffectMixin, QuantizedLatentMixin, TransitionAE):
#     pass
# class QuantizedDiffDetConditionalEffectTransitionAE(HammingMixin, ZeroSuppressMixin, DiffDetActionMixin, ConditionalEffectMixin, QuantizedLatentMixin, TransitionAE):
#     pass
# class QuantizedNonDetConditionalEffectTransitionAE (HammingMixin, ZeroSuppressMixin, NonDetActionMixin, ConditionalEffectMixin, QuantizedLatentMixin, TransitionAE):
#     pass
# 
# class SigmoidDetConditionalEffectTransitionAE    (HammingMixin, ZeroSuppressMixin, DetActionMixin, ConditionalEffectMixin, SigmoidLatentMixin, TransitionAE):
#     pass
# class SigmoidDiffDetConditionalEffectTransitionAE(HammingMixin, ZeroSuppressMixin, DiffDetActionMixin, ConditionalEffectMixin, SigmoidLatentMixin, TransitionAE):
#     pass
# class SigmoidNonDetConditionalEffectTransitionAE (HammingMixin, ZeroSuppressMixin, NonDetActionMixin, ConditionalEffectMixin, SigmoidLatentMixin, TransitionAE):
#     pass
# 
# class GumbelSoftmaxDetConditionalEffectTransitionAE    (HammingMixin, ZeroSuppressMixin, DetActionMixin, ConditionalEffectMixin, GumbelSoftmaxLatentMixin, TransitionAE):
#     pass
# class GumbelSoftmaxDiffDetConditionalEffectTransitionAE(HammingMixin, ZeroSuppressMixin, DiffDetActionMixin, ConditionalEffectMixin, GumbelSoftmaxLatentMixin, TransitionAE):
#     pass
# class GumbelSoftmaxNonDetConditionalEffectTransitionAE (HammingMixin, ZeroSuppressMixin, NonDetActionMixin, ConditionalEffectMixin, GumbelSoftmaxLatentMixin, TransitionAE):
#     pass
# 
# class SoftmaxDetConditionalEffectTransitionAE    (HammingMixin, ZeroSuppressMixin, DetActionMixin, ConditionalEffectMixin, SoftmaxLatentMixin, TransitionAE):
#     pass
# class SoftmaxDiffDetConditionalEffectTransitionAE(HammingMixin, ZeroSuppressMixin, DiffDetActionMixin, ConditionalEffectMixin, SoftmaxLatentMixin, TransitionAE):
#     pass
# class SoftmaxNonDetConditionalEffectTransitionAE (HammingMixin, ZeroSuppressMixin, NonDetActionMixin, ConditionalEffectMixin, SoftmaxLatentMixin, TransitionAE):
#     pass
