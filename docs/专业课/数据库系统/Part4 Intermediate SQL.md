# Part4: Intermediate SQL
## 1 Joined Relations
- A join operation is a Cartesian product which requires that tuples in the two relations match (under some condition)
- typically used as subquery expressions in the **from** clause
- types: natural join, inner join, outer join

### 1.1 natural join
- Natural join matches tuples with the same values for **all common attributes**, and retains only one copy of each common column.
```sql
select name, course_id
from students, takes
where students.id = takes.id
```
等价于
```sql
select name, course_id
from students natural join takes
```

!!! danger "Dangerous in Natural Join"
    be aware of unrelated attributes with same name which get equated incorrectly

    **To avoid**: use "using"

    ```sql
    select name, course_id
    from students natural join takes using (id)
    ```

- join condition:**on**
    ```sql
    select *
    from strudent join takes on student_ID = takes_ID
    ```

### 1.2 outer join
- Computes the join and then adds tuples form one relation that does not match tuples in the other relation to the result of the join. 
- use *null* values
- three forms: left outer join, right outer join, full outer join

## 2 Views
- Any relation that is not of the conceptual model but is made visible to a user as a “virtual relation” is called a view.(不是实体)
- a view definition causes the saving of an expression; the expression is substituted into queries using the view.

```sql
create view faculty as 
select ID, name, dept_name
from instructor
```
### 2.1 Views Defined Using Other Views
- depend directly
- depend on
- recursive (depends on itself)
### 2.2 Materialized Views
- physically stored
- If relations used in the query are updated, the materialized view result becomes out of date. **need to maintain the view**

### 2.3 Update of a View
- insert
- query: 
  - not have a **group by** or **having** clause

## 3 Index
!!! example ""
    create index studentID_index on student(ID)

query can be executed by using the index to find the required record,  without looking at all records of student

## 4 Transactions
- The transaction must end with one of the following statements:
  - commit work: updates becaome permanent
  - rollbck work: updates are undone
## 5 Integrity Constraints
### 5.1 on a single relation
1. not null
2. primary key
3. unique
4. check(P)
### 5.2 Referential integrity
!!! example ""
    foreign key (dept_name) references department (dept_name)

## 6 Complex Check Conditions
!!! example ""
    check (time_slot_id in (select time_slot_id from time_slot))

## 7 Assertions
!!! example ""
    create assertion <assertion-name> check (<predicate>);

## 8 Triggers
!!! hint "ECA rule"
    E: Event(insert, update, delete)
    C: Condition
    A: Action

    referencing old/new row as

!!! danger ""
    Risk of unintended execution of triggers

## 9 Build-in Data Types in SQL
- data: 2005-7-27
- time: 09:00:30.75
- timestamp: 2005-7-27 09:00:30.75
- interval: 1 day
- blob: binary large object
- clob: character large object

## 10 User-Defined Types
!!! example ""
    create type Dollars as numeric (12,2) final 

## 11 Domains
- Types and domains are similar.  Domains can have constraints, such as not null, specified on them.

!!! example ""
    create domain person_namechar(20) not null

## 12 Authorization(授权)
- forms of authorizations: read, insert, update, delete; index, resource, alteration, drop

!!! example ""
    grant <privilege list> on <relation or view> to <user list>

### 12.1 privileges in sql
- select
- insert
- update
- delete
- all privileges
### 12.2 revoking authorization
!!! example ""
    revoke <privilege list> on <relation or view> from <user list>

### 12.3 Roles
```sql
create role <name>

grant <role> to <user>
```
### 12.4 other authorization features
- grant reference (dept_name) on department to Mariano
- grant select on department to Amitwith grant option
- revoke select on department from Amit, Satoshi cascade
- revoke select on department from Amit, Satoshi restrict
















