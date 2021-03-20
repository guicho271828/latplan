select generators.name  as domain,
       kltune3."z1z2.mae.vanilla.test",
       nozsae3."z1z2.mae.vanilla.test",
       kltune4."z1z2.mae.vanilla.test",
       nozsae4."z1z2.mae.vanilla.test"
  from generators
         join (
           select generator, min("metrics.test.elbo") as elbo, "z1z2.mae.vanilla.test", beta_z, beta_d
             from ama3
            where comment in ('kltune2', 'notune2')
            group by generator
         ) as kltune3
             on generators.key = kltune3.generator
         join (
           select generator, min("metrics.test.elbo") as elbo, "z1z2.mae.vanilla.test", beta_z, beta_d
             from ama3
            where comment = 'nozsae2'
            group by generator
         ) as nozsae3
             on generators.key = nozsae3.generator
         join (
           select generator, min("metrics.test.elbo") as elbo, "z1z2.mae.vanilla.test", beta_z, beta_d
             from ama4
            where comment in ('kltune2', 'notune2')
            group by generator
         ) as kltune4
             on generators.key = kltune4.generator
         join (
           select generator, min("metrics.test.elbo") as elbo, "z1z2.mae.vanilla.test", beta_z, beta_d
             from ama4
            where comment = 'nozsae2'
            group by generator
         ) as nozsae4
             on generators.key = nozsae4.generator



