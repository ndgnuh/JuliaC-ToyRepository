trim: clean
	juliac  --output-exe main .  --experimental --trim
	# ./main ./config.toml

build:
	juliac  --output-exe main . --bundle 

clean:
	rm -rf bin lib share

run:
	juliap -e '\
	using RaylibCompileTest \
	RaylibCompileTest.main(String[])'
