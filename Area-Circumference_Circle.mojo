struct Circle:
    var pi: FloatLiteral
    var radius: FloatLiteral
    var area: FloatLiteral
    var circumference: FloatLiteral
    
def main():
    let circle01: circle
    
    circle01.pi = 3.1415
    circle01.radius = 20
    
    circle01.area = circle01.pi * (circle01.radius * circle01.radius)
    circle01.circumference = 2 * circle01.pi * circle01.radius
    print("Area:", circle01.area, "\nCircumference:", circle01.circumference)
    
main()
