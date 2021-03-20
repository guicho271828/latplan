
select generators.name  as domain,
       ama3.kl as "\ama3",
       ama4.kl as "\ama4"
  from generators
         join (
           select generator, min("metrics.test.elbo"),
                  "metrics.test.kl_a_z0" as kl
             from ama3
            where comment in ('kltune2', 'notune2')
            group by generator
         ) as ama3
             on generators.key = ama3.generator
         join (
           select generator, min("metrics.test.elbo"),
                  "metrics.test.kl_a_z0" as kl
             from ama4
            where comment in ('kltune2', 'notune2')
            group by generator
         ) as ama4
             on generators.key = ama4.generator
