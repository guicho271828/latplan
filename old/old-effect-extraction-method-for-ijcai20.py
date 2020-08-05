

# This method was used in the IJCAI20 paper.
# It uses the example transitions to extract the effect,
# thus has to loop over the dataset which is unnecessary slow.
# In Neurips2020 paper, this is replaced with a method
# that applies actions to just two states (zeros and ones).


        def extract_effect_from_transitions(transitions):
            pre = transitions[:,:N]
            suc = transitions[:,N:]
            data_diff = suc - pre
            data_add  = np.maximum(0, data_diff)
            data_del  = -np.minimum(0, data_diff)

            add_effect = np.zeros((true_num_actions, N))
            del_effect = np.zeros((true_num_actions, N))

            for i, a in enumerate(np.where(histogram > 0)[0]):
                indices = np.where(actions_byid == a)[0]
                add_effect[i] = np.amax(data_add[indices], axis=0)
                del_effect[i] = np.amax(data_del[indices], axis=0)

            return add_effect, del_effect, data_diff

        # effects obtained from the latent vectors
        add_effect2, del_effect2, diff2 = extract_effect_from_transitions(data)

        save("action_add2.csv",add_effect2)
        save("action_del2.csv",del_effect2)
        save("action_add2+ids.csv",np.concatenate((add_effect2,all_actions_byid), axis=1))
        save("action_del2+ids.csv",np.concatenate((del_effect2,all_actions_byid), axis=1))
        save("diff2+ids.csv",np.concatenate((diff2,actions_byid), axis=1))

        data_aae = np.concatenate([pre,self.decode_action([pre,actions], **kwargs)], axis=1)

        # effects obtained from the latent vectors, but the successor uses the ones coming from the AAE
        add_effect3, del_effect3, diff3 = extract_effect_from_transitions(data_aae)

        save("action_add3.csv",add_effect3)
        save("action_del3.csv",del_effect3)
        save("action_add3+ids.csv",np.concatenate((add_effect3,all_actions_byid), axis=1))
        save("action_del3+ids.csv",np.concatenate((del_effect3,all_actions_byid), axis=1))
        save("diff3+ids.csv",np.concatenate((diff3,actions_byid), axis=1))
