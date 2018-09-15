class Employee:
    #Class Variables
    empCount = 0
    totalSalary=0
    avgSalary=0

    #Constructor which will be called everytime a instance is created.
    def __init__(self, name, salary, family, department):
        self.name = name
        self.salary = salary
        self.family = family
        self.department = department
        #Class variable is incremented everytime a instace is created
        Employee.empCount += 1
        #Salary of each instance is added to class variable everytime a instace is created
        Employee.totalSalary += salary

    #Function to diplay number of employees
    def displayCount(self):
        print ("Total Employee %d" % Employee.empCount)

    #FUnction to display details of each instance.
    def displayEmployee(self):
        print ("Name : ", self.name, ", Salary: ", self.salary, " ,family: ",self.family, ", Department:",self.department,)

    # a function to average salary
    def CalculateAvg(self):
        Employee.avgSalary = Employee.totalSalary / Employee.empCount
        print ("Averegae salary is", Employee.avgSalary)


#Fulltime Employeeclass and it should inherit the properties of Employee class
class FullTimeEmp(Employee):
    VoucherValue=0
    #Constructor
    def __init__(self,name, salary, family, department,voucher):
        #Super Class constructor will be called
        Employee.__init__(self,name, salary, family, department)
        self.VoucherValue=voucher

#Creating instance of an Employee
emp = Employee("Mihir", 4000, "pitale", "chemistry")
#Creating instance of  subclass
fullemp = FullTimeEmp("Pallavi", 6000, "desai", "maths" ,5000)
# Calling functions
emp.displayEmployee()
emp.displayCount()
emp.CalculateAvg()
fullemp.displayEmployee()


