select generators.name  as domain,
       kltune.elbo   as ELBO,
       kltune.beta_z as "$\beta_1$",
       kltune.beta_d as "$\beta_3$",
       kltune.N      as "$F$",
       notune.elbo   as ELBO,
       notune.beta_z as "$\beta_1$",
       notune.beta_d as "$\beta_3$",
       notune.N      as "$F$",
       nozsae.elbo   as ELBO,
       nozsae.beta_z as "$\beta_1$",
       nozsae.beta_d as "$\beta_3$",
       nozsae.N      as "$F$"
  from generators
         join (
           select generator, min("metrics.test.elbo") as elbo, beta_z,beta_d,N
             from ama4
            where comment in ('kltune2', 'notune2')
            group by generator
         ) as kltune
             on generators.key = kltune.generator
         join (
           select generator, min("metrics.test.elbo") as elbo, beta_z,beta_d,N
             from ama4
            where comment = 'notune2'
            group by generator
         ) as notune
             on generators.key = notune.generator
         join (
           select generator, min("metrics.test.elbo") as elbo, beta_z,beta_d,N
             from ama4
            where comment = 'nozsae2'
            group by generator
         ) as nozsae
             on generators.key = nozsae.generator

