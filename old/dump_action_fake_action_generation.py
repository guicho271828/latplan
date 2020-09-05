
# this code was used originally by AD / SD to produce the fake dataset for PU-learning.
# The data generation code was subsequently moved to dump_action method.

# After the introduction of BTL, due to the obvious reason of avoiding the use of black-box preconditions,
# separately learning the AD is no longer necessary.
# The code may be useful in future journal paper, so it is kept here.

        def generate_aae_action(known_transisitons):
            states = known_transisitons.reshape(-1, N)
            from .util import set_difference
            def repeat_over(array, repeats, axis=0):
                array = np.expand_dims(array, axis)
                array = np.repeat(array, repeats, axis)
                return np.reshape(array,(*array.shape[:axis],-1,*array.shape[axis+2:]))
        
            print("start generating transitions")
            # s1,s2,s3,s1,s2,s3,....
            repeated_states  = repeat_over(states, len(all_labels), axis=0)
            # a1,a1,a1,a2,a2,a2,....
            repeated_actions = np.repeat(all_labels, len(states), axis=0)
        
            y = self.decode_action([repeated_states, repeated_actions], **kwargs).round().astype(np.int8)
            y = np.concatenate([repeated_states, y], axis=1)
        
            print("remove known transitions")
            y = set_difference(y, known_transisitons)
            print("shuffling")
            import numpy.random
            numpy.random.shuffle(y)
            return y
        
        transitions = generate_aae_action(data)
        # note: transitions are already shuffled, and also do not contain any examples in data.
        actions      = self.encode_action(transitions, **kwargs).round()
        actions_byid = to_id(actions)
        
        # ensure there are enough test examples
        separation = min(len(data)*10,len(transitions)-len(data))
        
        # fake dataset is used only for the training.
        fake_transitions  = transitions[:separation]
        fake_actions_byid = actions_byid[:separation]
        
        # remaining data are used only for the testing.
        test_transitions  = transitions[separation:]
        test_actions_byid = actions_byid[separation:]
        
        print(fake_transitions.shape, test_transitions.shape)
        
        save("fake_actions.csv",fake_transitions)
        save("fake_actions+ids.csv",np.concatenate((fake_transitions,fake_actions_byid), axis=1))
        
        from .util import puzzle_module
        p = puzzle_module(self.path)
        print("decoding pre")
        pre_images = self.decode(test_transitions[:,:N],**kwargs)
        print("decoding suc")
        suc_images = self.decode(test_transitions[:,N:],**kwargs)
        print("validating transitions")
        valid    = p.validate_transitions([pre_images, suc_images],**kwargs)
        invalid  = np.logical_not(valid)
        
        valid_transitions  = test_transitions [valid][:len(data)] # reduce the amount of data to reduce runtime
        valid_actions_byid = test_actions_byid[valid][:len(data)]
        invalid_transitions  = test_transitions [invalid][:len(data)] # reduce the amount of data to reduce runtime
        invalid_actions_byid = test_actions_byid[invalid][:len(data)]
        
        save("valid_actions.csv",valid_transitions)
        save("valid_actions+ids.csv",np.concatenate((valid_transitions,valid_actions_byid), axis=1))
        save("invalid_actions.csv",invalid_transitions)
        save("invalid_actions+ids.csv",np.concatenate((invalid_transitions,invalid_actions_byid), axis=1))
