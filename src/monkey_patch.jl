# Monkey patch packages to allow static compilation to work
module _Patches

using FixedPointNumbers: FixedPoint
import FixedPointNumbers

"""
Patch the FixedPointNumbers throw_converterror function to print to stderr and make static compilation works.
"""
@noinline function FixedPointNumbers.throw_converterror(
        ::Type{X}, x
    ) where {X <: FixedPoint}
    print(Core.stderr, "ConversionError: Cannot convert $x to $X\n")
    return nothing
end


macro Base.assert(condition, message)
    return quote
        if $(esc(condition))
        else
            println(Core.stderr, $(esc(message)))
            exit(1)
        end
    end
end

Base.print(x) = print(Core.stdout, x)
Base.println(x) = println(Core.stdout, x)

end
