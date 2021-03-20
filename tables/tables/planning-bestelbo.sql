
-- find the configuration that solved the most


with
  best as (
    select generator, comment, aeclass, time_start, min("metrics.test.elbo")
      from ama3
     group by generator, comment, aeclass
     union
    select generator, comment, aeclass, time_start, min("metrics.test.elbo")
      from ama4
     group by generator, comment, aeclass),
  solved as (
    select domain, heuristics, cycle, noise,
           "parameters.generator"  as generator,
           "parameters.comment"    as comment,
           "parameters.aeclass"    as aeclass,
           "parameters.time_start" as time_start,
           sum(found = "true") as found,
           sum(valid = "true") as valid,
           sum(valid = "true" and "statistics.plan_length" = cast(substr(problem, 1, 3) as int)) as optimal
      from planning
     group by domain, heuristics, cycle, noise, comment, aeclass, time_start)
select aeclass.name as aeclass,
       domains.name as domain,
       domains.id as domainid,
       comments.name as comment,
       heuristics.name as heuristics,
       cycle, noise, found, valid, optimal
  from best
         join solved
             on best.comment = solved.comment
             and best.generator = solved.generator
             and best.aeclass = solved.aeclass
             and best.time_start = solved.time_start
         join domains
             on domains.key = solved.domain
         join heuristics
             on heuristics.key = solved.heuristics
         join comments
             on comments.key = best.comment
         join aeclass
             on aeclass.key = best.aeclass
         
