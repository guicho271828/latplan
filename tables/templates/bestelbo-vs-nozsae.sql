
-- find the best elbo configuration


with
  best as (
    select generator, comment, aeclass, time_start, min("metrics.test.elbo")
      from AECLASS
     group by generator, comment, aeclass),
  statistics as (
    select heuristics,
           "parameters.generator"  as generator,
           "parameters.comment"    as comment,
           "parameters.aeclass"    as aeclass,
           "parameters.time_start" as time_start,
           domain, problem,
           ((valid = "true")*"statistics.generated")+((valid = "false")*(1000000000-random()%400000000))  as generated,
           ((valid = "true")*"statistics.expanded")+((valid = "false")*(1000000000-random()%400000000))  as expanded,
           ((valid = "true")*"statistics.evaluated")+((valid = "false")*(1000000000-random()%400000000))  as evaluated,
           ((valid = "true")*"statistics.search")+((valid = "false")*(10000-random()%2000))  as search
      from planning
     where noise = NOISE),
  best_stat as (
    select *
      from best join statistics
     where best.generator = statistics.generator
       and best.comment = statistics.comment
       and best.aeclass = statistics.aeclass
       and best.time_start = statistics.time_start),
  x as (
    select *
      from best_stat
     where comment = "kltune2"),
  y as (
    select *
      from best_stat
     where comment = "nozsae2")
select GROUP.id,
       x.METRIC, 0, y.METRIC, 0,
       x.problem, x.domain
  from x
         join y
             on  x.domain = y.domain
             and x.problem = y.problem
         join domains
             on x.domain = domains.key
         join generators
             on x.generator = generators.key
         join heuristics
             on x.heuristics = heuristics.key

