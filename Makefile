TARGET = ml

all: $(TARGET).c 
	gcc -o $(TARGET) $(TARGET).c -lm -lcrypto -lssl -lsodium

clean:
	rm $(TARGET)
