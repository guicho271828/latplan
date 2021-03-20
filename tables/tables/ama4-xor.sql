select generators.name  as domain,
       kltune."z1z2.xor.train" as "effects",
       kltune."z0z3.xor.train" as "precondition",
       kltune.N as "$F$",
       kltune."action.true_num_actions.train" as "$A_1$",
       kltune."z0z3.compiled_action.train" as "$A_2$"
  from generators
         join (
           select generator, min("metrics.test.elbo") as elbo, "z1z2.xor.train", "z0z3.xor.train", "action.true_num_actions.train", "z0z3.compiled_action.train", N
             from ama4
            where comment in ('kltune2', 'notune2')
            group by generator
         ) as kltune
             on generators.key = kltune.generator


-- select generators.name  as domain,
--        kltune.xor as xor,
--        kltune.a1 as "$A_1$",
--        kltune.a2 as "$A_2$"
--   from generators
--          join (
--            select generator, avg("z1z2.xor.train") as xor, avg("action.true_num_actions.train") as a1, avg("z1z2.compiled_action.train") as a2
--              from ama4
--             where comment in ('kltune2', 'notune2')
--             group by generator
--          ) as kltune
--              on generators.key = kltune.generator
