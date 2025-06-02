# Part 3: Introduntion to SQL
## 1 SQL Parts
###  1.1 DDL (Data-definition language)
#### Domain Types in SQL
- char(n): fixed length character string
- varchar(n): variable length character strings, with maximum length n
- int
- smallint
- numeric(p, d): fixed point number with p digits and d decimal point

    !!! example ""
        numeric(3, 1)  44.5

- real, doubke precision: machine-dependent precision
- float(n): user-specified precision of at least n digits

#### Create Table Construct
```sql
create tableinstructor(
    ID  char(5),
    name varchar(20),
    dept_name  varchar(20),
    salary  numeric(8,2))
```
#### Integrity Constraints in Create Table
```sql
primary key(a, ...,b)
foreign key(a, ...,b) references r (on delete/update cascate | set null | restrict | set default)
not null
```
### 1.2 DML (Data-Manipulation language)
#### Updates to tables
- insert

    !!! example ""
        insert into instructor values('1001', 'John', 'Computer Science', 50000)

- delete

    !!! example ""
        delete from student

- drop table

    !!! example ""
        drop table instructor

- alter

    !!! example ""
        alter table r add A D

        alter table r drop A

#### Basic Query Structure
```sql
select (distinct/all) $A_1, A_2,...,A_n$ 默认不去重
from $r_1, r_2, ..., r_m$
where P
```
- result of an SQL query is a relation
- The selectclause can contain arithmetic expressions involving the operation, +, –, , and /：`selectID, name, salary/12`

#### String Operations
- %: The % character matches any substring
- _: The _character matches any character

!!! example ""
    ```sql
    select name
    from instructor
    where name like '%dar%'

    like '100\%' escape '\' //查询100%
    ```

- supports:
  - concatenation ("||")
  - converting from upper to lower case ("lower()") and vice versa ("upper()")
  - finding string length, extracting substrings, etc.

#### Ordering the Display of Tuples

!!! example ""
    order by name desc/asc //默认asc（升序）

#### Set Operations

!!! warning ""
    默认去重，若不需要：union all

- union: 并
- intersect: 交
- except: 差

#### NULL Values
null 被视为未知量，如 true and unknown = unknown

#### Aggregate Functions
- avg
- min
- max
- sum
- count: number of values

    !!! example "aggregate functions"
        ```sql
        select avg(salary)
        from instructor
        where dept_name = 'Comp.Sci';
        ```

- group by

    !!! example ""
        select dept_name, avg(salary) as avg_salary
        from instructor
        group by dept_name;
        having avg(salary) > 42000;

#### Set Membership
- in / not in

    !!! example ""
        ```sql
        select distinct course_id
        from section
        where semester = 'Fall and year = 2017 and
        course_id in (select course_id from section where semester = 'Spring and year = 2018');
        ```
#### Set Comparison
```sql
select distinct T.name
from instructor as T, instructor as S
where T.salary > S.salary and S.dept_name = 'Biology';
```
等价于
```sql
selsct name
from instructor 
where salary > some(select salary
                    from instructor
                    where dept_name = 'Biology');
```



- all: 比任何一个都...

  !!! hint "some"
        =some $\equiv$ in
        $\neq$some $\not \equiv$not in

        $\neq$all $\equiv$ not in
        =all $\not \equiv$ in

- exists / not exists: The exists construct returns the value true if the argument subquery is nonempty / empty.
```sql
where not exists(...)
```
- unipue: evaluates to “true” if a given subquery contains no duplicates 

#### Subqueries in the From Clause
- With Clause(定义临时表)

    !!! example ""
        ```sql
        with max_budget(value) as
        (select max(budget) from deprtment)
        select department.name
        from department, max_bugdet
        where department.budget != max_bugdet.value;
        ```

- Scalar Subquery
    - a single value is expected

#### Modification of the Database
- deletion

    !!! example ""
    ```sql
    delete from instructor
    ```

- insertion

    !!! example ""
    ```sql
    insert into instructor values('1001', 'John', 'Computer Science', 50000)
    ```

- update

    !!! example ""
    ```sql
    update instructor set salary = 55000 where ID = '1001'
    ```

!!! danger "Case Statement for Conditional Updates"
    ```sql
    update instructor
    set salary = case
                    when salary <= 100000 then salary * 1.05
                    else salary * 1.03
                end
    ```