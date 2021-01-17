TARGET = ml

all: $(TARGET).c 
	gcc -o $(TARGET) $(TARGET).c -lm -lcrypto -lssl

clean:
	rm $(TARGET)
