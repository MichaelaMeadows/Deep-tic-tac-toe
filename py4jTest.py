from py4j.java_gateway import JavaGateway
gateway = JavaGateway()
#random = gateway.jvm.java.util.Random()   # create a java.util.Random instance
#number1 = random.nextInt(10)

gateway.innitGame()
state = gateway.getState()
print(state[0])
