
/* path_parse_svg.cpp - Boost Spirit based parser for SVG path grammar */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#if 0
#define BOOST_SPIRIT_DEBUG
#endif

#ifdef _MSC_VER 
#pragma warning( disable:4244 ) // '=' : conversion from 'int' to 'float', possible loss of data
#endif

#include <boost/spirit/include/classic.hpp>
#include <boost/spirit/include/classic_push_back_actor.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <boost/function.hpp>
#include <boost/bind.hpp>

#include "countof.h"
 
using namespace std;
using namespace boost::spirit::classic;

// Boost Spirit parser for the SVG 1.1 path grammar
// http://dev.w3.org/SVG/profiles/1.1F2/publish/paths.html#PathDataBNF
struct svg_parse_data {
    svg_parse_data(vector<char> &c, vector<float> &v) : commands(c), vertexes(v) {
        c.clear();
        v.clear();
    }

    vector<char> &commands;
    vector<float> &vertexes;
    char last_command;
};

// ureal_p & real_p parse real number strings into doubles; we just want floats.
real_parser<float, ureal_parser_policies<float> > const
    ufloat_p     = real_parser<float, ureal_parser_policies<float> >();
real_parser<float, real_parser_policies<float> > const
    float_p      = real_parser<float, real_parser_policies<float> >();

static const char ch_l = 'l', ch_L = 'L';

struct svg_grammar : public grammar<svg_grammar> {
    svg_grammar(svg_parse_data &_data)
        : data(_data)
    {}

    static const float f_zero, f_one;

    template <typename ScannerT>
    struct definition {
        definition(const svg_grammar &self)
            : data(self.data)
        {
            //svg-path:
            //    wsp* moveto-drawto-command-groups? wsp*
            svg_path
                = *wsp
                  >> !moveto_drawto_command_groups
                  >> *wsp;
            //moveto-drawto-command-groups:
            //    moveto-drawto-command-group
            //    | moveto-drawto-command-group wsp* moveto-drawto-command-groups
            moveto_drawto_command_groups
                = moveto_drawto_command_group
                  >> *(*wsp >> moveto_drawto_command_group);
            //moveto-drawto-command-group:
            //    moveto wsp* drawto-commands?
            moveto_drawto_command_group
                = moveto
                  >> *wsp
                  >> !drawto_commands;
            //drawto-commands:
            //    drawto-command
            //    | drawto-command wsp* drawto-commands
            drawto_commands
                = drawto_command
                  >> *(*wsp >> drawto_command);
            //drawto-command:
            //    closepath
            //    | lineto
            //    | horizontal-lineto
            //    | vertical-lineto
            //    | curveto
            //    | smooth-curveto
            //    | quadratic-bezier-curveto
            //    | smooth-quadratic-bezier-curveto
            //    | elliptical-arc
            drawto_command
                = lineto
                | horizontal_lineto
                | vertical_lineto
                | curveto
                | smooth_curveto
                | quadratic_bezier_curveto
                | smooth_quadratic_bezier_curveto
                | elliptical_arc
                | closepath;
            //moveto:
            //    ( "M" | "m" ) wsp* moveto-argument-sequence
            moveto
                = ( ch_p('M')[push_back_a(data.commands)][assign_a(data.last_command, ch_L)] |
                    ch_p('m')[push_back_a(data.commands)][assign_a(data.last_command, ch_l)] )
                  >> *wsp
                  >> moveto_argument_sequence;
            //moveto-argument-sequence:
            //    coordinate-pair
            //    | coordinate-pair comma-wsp? lineto-argument-sequence
            moveto_argument_sequence
                = coordinate_pair
                  >> *(!comma_wsp >> lineto_argument_sequence[push_back_a(data.commands,data.last_command)]);
            //closepath:
            //    ("Z" | "z")
            closepath
                = ch_p('Z')[push_back_a(data.commands)]
                | ch_p('z')[push_back_a(data.commands)];
            //lineto:
            //    ( "L" | "l" ) wsp* lineto-argument-sequence
            lineto
                = ( ch_p('L')[push_back_a(data.commands)][assign_a(data.last_command)] |
                    ch_p('l')[push_back_a(data.commands)][assign_a(data.last_command)] )
                  >> *wsp
                  >> lineto_argument_sequence;
            //lineto-argument-sequence:
            //    coordinate-pair
            //    | coordinate-pair comma-wsp? lineto-argument-sequence
            lineto_argument_sequence
                = coordinate_pair
                  >> *(!comma_wsp >> coordinate_pair[push_back_a(data.commands,data.last_command)]);
            //horizontal-lineto:
            //    ( "H" | "h" ) wsp* horizontal-lineto-argument-sequence
            horizontal_lineto
                = ( ch_p('H')[push_back_a(data.commands)][assign_a(data.last_command)] |
                    ch_p('h')[push_back_a(data.commands)][assign_a(data.last_command)] )
                  >> *wsp
                  >> horizontal_lineto_argument_sequence;
            //horizontal-lineto-argument-sequence:
            //    coordinate
            //    | coordinate comma-wsp? horizontal-lineto-argument-sequence
            horizontal_lineto_argument_sequence
                = coordinate
                  >> *(!comma_wsp >> coordinate[push_back_a(data.commands,data.last_command)]);
            //vertical-lineto:
            //    ( "V" | "v" ) wsp* vertical-lineto-argument-sequence
            vertical_lineto
                = ( ch_p('V')[push_back_a(data.commands)][assign_a(data.last_command)] |
                    ch_p('v')[push_back_a(data.commands)][assign_a(data.last_command)] )
                  >> *wsp
                  >> vertical_lineto_argument_sequence;
            //vertical-lineto-argument-sequence:
            //    coordinate
            //    | coordinate comma-wsp? vertical-lineto-argument-sequence
            vertical_lineto_argument_sequence
                = coordinate
                  >> *(!comma_wsp >> coordinate[push_back_a(data.commands,data.last_command)]);
            //curveto:
            //    ( "C" | "c" ) wsp* curveto-argument-sequence
            curveto
                = ( ch_p('C')[push_back_a(data.commands)][assign_a(data.last_command)] |
                    ch_p('c')[push_back_a(data.commands)][assign_a(data.last_command)] )
                  >> *wsp
                  >> curveto_argument_sequence;
            //curveto-argument-sequence:
            //    curveto-argument
            //    | curveto-argument comma-wsp? curveto-argument-sequence
            curveto_argument_sequence
                = curveto_argument
                  >> *(!comma_wsp >> curveto_argument[push_back_a(data.commands,data.last_command)]);
            //curveto-argument:
            //    coordinate-pair comma-wsp? coordinate-pair comma-wsp? coordinate-pair
            curveto_argument
                = coordinate_pair
                  >> !comma_wsp
                  >> coordinate_pair
                  >> !comma_wsp
                  >> coordinate_pair;
            //smooth-curveto:
            //    ( "S" | "s" ) wsp* smooth-curveto-argument-sequence
            smooth_curveto
                = ( ch_p('S')[push_back_a(data.commands)][assign_a(data.last_command)] |
                    ch_p('s')[push_back_a(data.commands)][assign_a(data.last_command)] )
                  >> *wsp
                  >> smooth_curveto_argument_sequence;
            //smooth-curveto-argument-sequence:
            //    smooth-curveto-argument
            //    | smooth-curveto-argument comma-wsp? smooth-curveto-argument-sequence
            smooth_curveto_argument_sequence
                = smooth_curveto_argument
                  >> *(!comma_wsp >> smooth_curveto_argument[push_back_a(data.commands,data.last_command)]);
            //smooth-curveto-argument:
            //    coordinate-pair comma-wsp? coordinate-pair
            smooth_curveto_argument
                = coordinate_pair
                  >> !comma_wsp
                  >> coordinate_pair;
            //quadratic-bezier-curveto:
            //    ( "Q" | "q" ) wsp* quadratic-bezier-curveto-argument-sequence
            quadratic_bezier_curveto
                = ( ch_p('Q')[push_back_a(data.commands)][assign_a(data.last_command)] |
                    ch_p('q')[push_back_a(data.commands)][assign_a(data.last_command)] )
                  >> *wsp
                  >> quadratic_bezier_curveto_argument_sequence;
            //quadratic-bezier-curveto-argument-sequence:
            //    quadratic-bezier-curveto-argument
            //    | quadratic-bezier-curveto-argument comma-wsp? 
            //        quadratic-bezier-curveto-argument-sequence
            quadratic_bezier_curveto_argument_sequence
                = quadratic_bezier_curveto_argument
                  >> *(!comma_wsp >> quadratic_bezier_curveto_argument[push_back_a(data.commands,data.last_command)]);
            //quadratic-bezier-curveto-argument:
            //    coordinate-pair comma-wsp? coordinate-pair
            quadratic_bezier_curveto_argument
                = coordinate_pair
                  >> !comma_wsp
                  >> coordinate_pair;
            //smooth-quadratic-bezier-curveto:
            //    ( "T" | "t" ) wsp* smooth-quadratic-bezier-curveto-argument-sequence
            smooth_quadratic_bezier_curveto
                = ( ch_p('T')[push_back_a(data.commands)][assign_a(data.last_command)] |
                    ch_p('t')[push_back_a(data.commands)][assign_a(data.last_command)] )
                  >> *wsp
                  >> smooth_quadratic_bezier_curveto_argument_sequence;
            //smooth-quadratic-bezier-curveto-argument-sequence:
            //    coordinate-pair
            //    | coordinate-pair comma-wsp? smooth-quadratic-bezier-curveto-argument-sequence
            smooth_quadratic_bezier_curveto_argument_sequence
                = coordinate_pair
                  >> *(!comma_wsp >> coordinate_pair[push_back_a(data.commands,data.last_command)]);
            //elliptical-arc:
            //    ( "A" | "a" ) wsp* elliptical-arc-argument-sequence
            elliptical_arc
                = ( ch_p('A')[push_back_a(data.commands)][assign_a(data.last_command)] |
                    ch_p('a')[push_back_a(data.commands)][assign_a(data.last_command)] )
                  >> *wsp
                  >> elliptical_arc_argument_sequence;
            //elliptical-arc-argument-sequence:
            //    elliptical-arc-argument
            //    | elliptical-arc-argument comma-wsp? elliptical-arc-argument-sequence
            elliptical_arc_argument_sequence
                = elliptical_arc_argument
                  >> *(!comma_wsp >> elliptical_arc_argument[push_back_a(data.commands,data.last_command)]);
            //elliptical-arc-argument:
            //    nonnegative-number comma-wsp? nonnegative-number comma-wsp? 
            //        number comma-wsp flag comma-wsp flag comma-wsp coordinate-pair
            elliptical_arc_argument
                = nonnegative_number
                  >> !comma_wsp
                  >> nonnegative_number
                  >> !comma_wsp
                  >> number
                  >> comma_wsp
                  >> flag
                  >> comma_wsp
                  >> flag
                  >> comma_wsp
                  >> coordinate_pair;
            //coordinate-pair:
            //    coordinate comma-wsp? coordinate
            coordinate_pair
                = coordinate
                  >> !comma_wsp
                  >> coordinate;
            //coordinate:
            //    number
            coordinate
                = number;
            //nonnegative-number:
            //    integer-constant
            //    | floating-point-constant
            nonnegative_number
                = ufloat_p[push_back_a(data.vertexes)];
            //number:
            //    sign? integer-constant
            //    | sign? floating-point-constant
            number
                = float_p[push_back_a(data.vertexes)];
            //flag:
            //    "0" | "1"
            flag
                //= range_p('0', '1')[push_back_a(data.vertexes)];
                = ch_p('0')[push_back_a(data.vertexes, f_zero)]
                | ch_p('1')[push_back_a(data.vertexes, f_one)];
            //comma-wsp:
            //    (wsp+ comma? wsp*) | (comma wsp*)
            comma_wsp
                = (+wsp >> !comma >> *wsp)
                | (comma >> *wsp);
            comma
                = ch_p(',');
            //integer-constant:
            //    digit-sequence
            //floating-point-constant:
            //    fractional-constant exponent?
            //    | digit-sequence exponent
            //fractional-constant:
            //    digit-sequence? "." digit-sequence
            //    | digit-sequence "."
            //exponent:
            //    ( "e" | "E" ) sign? digit-sequence
            //sign:
            //    "+" | "-"
            //digit-sequence:
            //    digit
            //    | digit digit-sequence
            //digit:
            //    "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
            //wsp:
            //    (#x20 | #x9 | #xD | #xA)
            wsp
                = space_p;

            BOOST_SPIRIT_DEBUG_RULE(svg_path);
            BOOST_SPIRIT_DEBUG_RULE(moveto_drawto_command_groups);
            BOOST_SPIRIT_DEBUG_RULE(moveto_drawto_command_group);
            BOOST_SPIRIT_DEBUG_RULE(drawto_commands);
            BOOST_SPIRIT_DEBUG_RULE(drawto_command);
            BOOST_SPIRIT_DEBUG_RULE(moveto);
            BOOST_SPIRIT_DEBUG_RULE(moveto_argument_sequence);
            BOOST_SPIRIT_DEBUG_RULE(closepath);
            BOOST_SPIRIT_DEBUG_RULE(lineto);
            BOOST_SPIRIT_DEBUG_RULE(lineto_argument_sequence);
            BOOST_SPIRIT_DEBUG_RULE(horizontal_lineto);
            BOOST_SPIRIT_DEBUG_RULE(horizontal_lineto_argument_sequence);
            BOOST_SPIRIT_DEBUG_RULE(vertical_lineto);
            BOOST_SPIRIT_DEBUG_RULE(vertical_lineto_argument_sequence);
            BOOST_SPIRIT_DEBUG_RULE(curveto);
            BOOST_SPIRIT_DEBUG_RULE(curveto_argument_sequence);
            BOOST_SPIRIT_DEBUG_RULE(curveto_argument);
            BOOST_SPIRIT_DEBUG_RULE(smooth_curveto);
            BOOST_SPIRIT_DEBUG_RULE(smooth_curveto_argument_sequence);
            BOOST_SPIRIT_DEBUG_RULE(smooth_curveto_argument);
            BOOST_SPIRIT_DEBUG_RULE(quadratic_bezier_curveto);
            BOOST_SPIRIT_DEBUG_RULE(quadratic_bezier_curveto_argument_sequence);
            BOOST_SPIRIT_DEBUG_RULE(quadratic_bezier_curveto_argument);
            BOOST_SPIRIT_DEBUG_RULE(smooth_quadratic_bezier_curveto);
            BOOST_SPIRIT_DEBUG_RULE(smooth_quadratic_bezier_curveto_argument_sequence);
            BOOST_SPIRIT_DEBUG_RULE(elliptical_arc);
            BOOST_SPIRIT_DEBUG_RULE(elliptical_arc_argument_sequence);
            BOOST_SPIRIT_DEBUG_RULE(coordinate_pair);
            BOOST_SPIRIT_DEBUG_RULE(coordinate);
            BOOST_SPIRIT_DEBUG_RULE(nonnegative_number);
            BOOST_SPIRIT_DEBUG_RULE(number);
            BOOST_SPIRIT_DEBUG_RULE(flag);
            BOOST_SPIRIT_DEBUG_RULE(comma_wsp);
            BOOST_SPIRIT_DEBUG_RULE(comma);
            BOOST_SPIRIT_DEBUG_RULE(integer_constant);
            BOOST_SPIRIT_DEBUG_RULE(floating_point_constant);
            BOOST_SPIRIT_DEBUG_RULE(fractional_constant);
            BOOST_SPIRIT_DEBUG_RULE(exponent);
            BOOST_SPIRIT_DEBUG_RULE(sign);
            BOOST_SPIRIT_DEBUG_RULE(digit_sequence);
            BOOST_SPIRIT_DEBUG_RULE(digit);
            BOOST_SPIRIT_DEBUG_RULE(wsp);
        }

        rule<ScannerT> const& start() const { return svg_path; }
        rule<ScannerT> svg_path,
                       moveto_drawto_command_groups,
                       moveto_drawto_command_group,
                       drawto_commands,
                       drawto_command,
                       moveto,
                       moveto_argument_sequence,
                       closepath,
                       lineto,
                       lineto_argument_sequence,
                       horizontal_lineto,
                       horizontal_lineto_argument_sequence,
                       vertical_lineto,
                       vertical_lineto_argument_sequence,
                       curveto,
                       curveto_argument_sequence,
                       curveto_argument,
                       smooth_curveto,
                       smooth_curveto_argument_sequence,
                       smooth_curveto_argument,
                       quadratic_bezier_curveto,
                       quadratic_bezier_curveto_argument_sequence,
                       quadratic_bezier_curveto_argument,
                       smooth_quadratic_bezier_curveto,
                       smooth_quadratic_bezier_curveto_argument_sequence,
                       elliptical_arc,
                       elliptical_arc_argument_sequence,
                       elliptical_arc_argument,
                       coordinate_pair,
                       coordinate,
                       nonnegative_number,
                       number,
                       flag,
                       comma_wsp,
                       comma,
                       integer_constant,
                       floating_point_constant,
                       fractional_constant,
                       exponent,
                       sign,
                       digit_sequence,
                       digit,
                       wsp;

        svg_parse_data &data; 
    };

    svg_parse_data &data; 
};

const float svg_grammar::f_zero = 0, svg_grammar::f_one = 1;

int spirit_svg_path_parser(const char *input, vector<char> &c, vector<float> &v)
{
    svg_parse_data data(c,v);

    svg_grammar svg(data);
    parse_info<> result = parse(input, svg);
    int status = result.full;
#if 0  // debugging code
    cout << status << endl;
    cout << input << endl;

    vector<float>::iterator d = data.vertexes.begin();
    for (size_t i=0; i<data.commands.size(); i++) {
        cout << data.commands[i] << ": ";
        switch (data.commands[i]) {
        case 'z':
        case 'Z':
            cout << endl;
            break;
        case 'm':
        case 'M':
        case 'L':
        case 'l':
        case 't':
        case 'T':
            cout << d[0] << "," << d[1] << endl;
            d += 2;
            break;
        case 'a':
        case 'A':
            cout << "radii=" << d[0] << "," << d[1]
                 << " x-axis-rotate=" << d[2]
                 << " large-arc-flag=" << d[3]
                 << " sweep-flag=" << d[4]
                 << " " << d[5] << "," << d[6] << endl;
            d += 7;
            break;
        case 'c':
        case 'C':
            cout << d[0] << "," << d[1] << " "
                 << d[2] << "," << d[3] << " "
                 << d[4] << "," << d[5] << endl;
            d += 6;
            break;
        case 'q':
        case 'Q':
        case 's':
        case 'S':
            cout << d[0] << "," << d[1] << " "
                 << d[2] << "," << d[3] << endl;
            d += 4;
            break;
        case 'v':
        case 'V':
        case 'h':
        case 'H':
            cout << d[0] << endl;
            d += 1;
            break;
        default:
            assert(!"unexpected command");
            break;
        }
    }
    assert(d == data.vertexes.end());
#endif
    return status;
}

// Parsers derive from this class to hide direct access to the string being parsed.
template <typename T>
class LimitedStringAccess {
private:
    const T *input;
    const T *end;
    const T *cur;
    T cur_char;
protected:
    inline T curChar() {
        return cur_char;
    }
    inline void updateCurChar() {
        if (cur < end) {
            assert(cur >= input);
            cur_char = *cur;
        } else {
            // Beyond the input string.
            cur_char = 0;  // some illegal character in the grammar
        }
    }
    inline void advanceChar() {
        cur++;
        updateCurChar();
    }

    // For back-tracking
    inline void rewindToPosition(const void *save) {
        assert(save >= input);
        assert(save <= end);
        cur = reinterpret_cast<const T*>(save);
        updateCurChar();
    }
    inline const void *savePosition() {
        return cur;
    }

    inline bool acceptChar(T c) {
        if (c == cur_char) {
            advanceChar();
            return true;
        }
        return false;
    }
    inline bool expectChar(T c) {
        return cur_char == c;
    }

    inline bool atEnd() {
        if (cur == end) {
            assert(cur_char == 0);
            return true;
        } else {
            return false;
        }
    }

    inline int errorPosition() {
        if (cur == end) {
            return -1;
        } else {
            assert(cur >= input);
            assert(end >= cur);
            return cur - input;
        }
    }

    LimitedStringAccess(const T *start, const T *end_)
        : input(start)
        , end(end_)
        , cur(start)
        , cur_char(*cur)
    { }
};

// Recursive descent parser for the SVG 1.1 path grammar
// http://dev.w3.org/SVG/profiles/1.1F2/publish/paths.html#PathDataBNF
class SVGPathParser : LimitedStringAccess<char> {
    vector<char> &cmd;
    vector<float> &coord;
    char current_command;

    inline bool acceptChars(char cmd1, char cmd2) {
        return acceptChar(cmd1) || acceptChar(cmd2);
    }

    inline void expectCmds(char cmd1, char cmd2) {
        assert(expectChar(cmd1) || expectChar(cmd2));
        current_command = curChar();
        advanceChar();
    }

    //wsp:
    //    (#x20 | #x9 | #xD | #xA)
    bool wsp() {
        switch (curChar()) {
        case 0x20:  // space
        case 0x9:   // tab
        case 0xD:   // return
        case 0xA:   // newline
            advanceChar();
            return true;
        default:
            return false;
        }
    }

    // wsp*
    void wsp_star() {
        while (wsp()) {
        }
    }

    // sign:
    //     "+" | "-"
    bool sign(int &number_sign) {
        if (acceptChar('-')) {
            number_sign = -1;
            return true;
        }
        if (acceptChar('+')) {
            number_sign = 1;
            return true;
        }
        return false;
    }

    // digit:
    //     "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    bool digit(int &digit_value) {
        if (curChar() >= '0' && curChar() <= '9') {
            digit_value = curChar() - '0';
            advanceChar();
            return true;
        }
        return false;
    }

    // digit-sequence:
    //     digit
    //     | digit digit-sequence
    bool digit_sequence(double &value) {
        int digit_value;
        if (!digit(digit_value)) {
            return false;
        }
        value = digit_value;
        while (digit(digit_value)) {
            value *= 10;
            value += digit_value;
        }
        return true;
    }
    // Use this when the digit-sequence is for a fractional value...
    bool fractional_digit_sequence(double &value) {
        int digit_value;
        if (!digit(digit_value)) {
            return false;
        }
        int digits = 1, effective_digits=1;
        value = digit_value;
        double effective_value = digit_value;
        while (digit(digit_value)) {
            value *= 10;
            value += digit_value;
            digits++;
            if (digit_value > 0) {
                // To ignore trailing zeros.
                effective_digits = digits;
                effective_value = value;
            }
        }
        value = effective_value/pow(10.0, effective_digits);
        return true;
    }

    // integer-constant:
    //     digit-sequence
    inline bool integer_constant(double &value) {
        return digit_sequence(value);
    }

    // fractional-constant:
    //     digit-sequence? "." digit-sequence
    //     | digit-sequence "."
    bool fractional_constant(double &value) {
        const void *save = savePosition();
        value = 0;
        if (!digit_sequence(value)) {
            rewindToPosition(save);
        }
        if (acceptChar('.')) {
            double fractional_value;
            if (fractional_digit_sequence(fractional_value)) {
                value += fractional_value;
                return true;
            }
        }
        rewindToPosition(save);
        value = 0;
        if (!digit_sequence(value)) {
            return false;
        }
        if (acceptChar('.')) {
            return true;
        }
        rewindToPosition(save);
        return false;
    }

    // exponent:
    //     ( "e" | "E" ) sign? digit-sequence
    bool exponent(double &exponent_value) {
        if (!acceptChars('e', 'E')) {
            return false;
        }
        const void *save = savePosition();
        int number_sign = 1;
        if (!sign(number_sign)) {
            rewindToPosition(save);
        }
        if (!digit_sequence(exponent_value)) {
            return false;
        }
        exponent_value *= number_sign;
        return true;
    }

    // floating-point-constant:
    //     fractional-constant exponent?
    //     | digit-sequence exponent
    bool floating_point_constant(double &value) {
        const void *save = savePosition();
        double exponent_value;
        if (fractional_constant(value)) {
            const void *save2 = savePosition();
            if (!exponent(exponent_value)) {
                rewindToPosition(save2);
            } else {
                value *= pow(10, exponent_value);
            }
            return true;
        }
        rewindToPosition(save);
        if (digit_sequence(value)) {
            const void *save2 = savePosition();
            if (!exponent(exponent_value)) {
                rewindToPosition(save2);
                return false;
            }
            value *= pow(10, exponent_value);
            return true;
        }
        rewindToPosition(save);
        return false;
    }

    // number:
    //     sign? integer-constant
    //     | sign? floating-point-constant
    bool number(double &value) {
        const void *save = savePosition();
        int number_sign = 1;
        if (!sign(number_sign)) {
            rewindToPosition(save);
        }
        // NOTE: match floating-point-constant ahead of integer-constant
        save = savePosition();
        if (floating_point_constant(value)) {
            value *= number_sign;
            return true;
        }
        rewindToPosition(save);
        if (integer_constant(value)) {
            value *= number_sign;
            return true;
        }
        rewindToPosition(save);
        return false;
    }

    // coordinate:
    //     number
    inline bool coordinate(double &value) {
        return number(value);
    }

    // comma:
    //     ","
    inline bool comma() {
        return acceptChar(',');
    }

    // comma-wsp:
    //     (wsp+ comma? wsp*) | (comma wsp*)
    bool comma_wsp_opt() {
        const void *save = savePosition();
        if (comma()) {
            wsp_star();
            return true;
        }
        rewindToPosition(save);
        if (wsp()) {
            wsp_star();
            const void *save2 = savePosition();
            if (!comma()) {
                rewindToPosition(save2);
            }
            wsp_star();
            return true;
        }
        rewindToPosition(save);
        return false;
    }

    // coordinate-pair:
    //     coordinate comma-wsp? coordinate
    bool coordinate_pair(double &x, double &y) {
        const void *save = savePosition();
        if (!coordinate(x)) {
            rewindToPosition(save);
            return false;
        }
        comma_wsp_opt();
        if (!coordinate(y)) {
            rewindToPosition(save);
            return false;
        }
        return true;
    }

    // lineto-argument-sequence:
    //     coordinate-pair
    //     | coordinate-pair comma-wsp? lineto-argument-sequence
    bool lineto_argument_sequence() {
        const void *save = savePosition();
        double x, y;
        if (!coordinate_pair(x, y)) {
            rewindToPosition(save);
            return false;
        }
        do {
            cmd.push_back(current_command);
            coord.push_back(float(x));
            coord.push_back(float(y));

            save = savePosition();
            comma_wsp_opt();
        } while(coordinate_pair(x, y));
        rewindToPosition(save);
        return true;
    }

    // moveto-argument-sequence:
    //    coordinate-pair
    //    | coordinate-pair comma-wsp? lineto-argument-sequence
    bool moveto_argument_sequence() {
        const void *save = savePosition();
        double x, y;
        if (!coordinate_pair(x, y)) {
            rewindToPosition(save);
            return false;
        }
        cmd.push_back(current_command);
        coord.push_back(float(x));
        coord.push_back(float(y));

        const void *save2 = savePosition();
        comma_wsp_opt();
        /* SVG 1.1 2nd Ed. "If a moveto is followed by multiple pairs of coordinates,
           the subsequent pairs are treated as implicit lineto commands.
           Hence, implicit lineto commands will be relative if the moveto
           is relative, and absolute if the moveto is absolute. */
        if (current_command == 'M') {
            current_command = 'L';
        } else {
            current_command = 'l';
        }
        if (!lineto_argument_sequence()) {
            rewindToPosition(save2);
        }
        return true;
    }

    // moveto:
    //     ( "M" | "m" ) wsp* moveto-argument-sequence
    bool moveto() {
        const void *save = savePosition();

        if (curChar() != 'M' && curChar() != 'm') {
            rewindToPosition(save);
            return false;
        }
        current_command = curChar();
        advanceChar();
        wsp_star();
        if (!moveto_argument_sequence()) {
            rewindToPosition(save);
            return false;
        }
        return true;
    }

    // closepath:
    //     ("Z" | "z")
    bool closepath() {
        expectCmds('z', 'Z');
        cmd.push_back(current_command);
        return true;
    }

    // lineto:
    //     ( "L" | "l" ) wsp* lineto-argument-sequence
    bool lineto() {
        const void *save = savePosition();
        expectCmds('l', 'L');
        wsp_star();
        if (!lineto_argument_sequence()) {
            rewindToPosition(save);
            return false;
        }
        return true;
    }

    // horizontal-lineto-argument-sequence:
    //     coordinate
    //     | coordinate comma-wsp? horizontal-lineto-argument-sequence
    bool horizontal_lineto_argument_sequence() {
        const void *save = savePosition();
        double x;
        if (!coordinate(x)) {
            rewindToPosition(save);
            return false;
        }
        do {
            cmd.push_back(current_command);
            coord.push_back(float(x));

            save = savePosition();
            comma_wsp_opt();
        } while(coordinate(x));
        rewindToPosition(save);
        return true;
    }

    // horizontal-lineto:
    //     ( "H" | "h" ) wsp* horizontal-lineto-argument-sequence
    bool horizontal_lineto() {
        const void *save = savePosition();
        expectCmds('h', 'H');
        wsp_star();
        if (!horizontal_lineto_argument_sequence()) {
            rewindToPosition(save);
            return false;
        }
        return true;
    }

    // vertical-lineto-argument-sequence:
    //     coordinate
    //     | coordinate comma-wsp? vertical-lineto-argument-sequence
    bool vertical_lineto_argument_sequence() {
        // Same rule for horizontal and vertical...
        return horizontal_lineto_argument_sequence();
    }

    // vertical-lineto:
    //     ( "V" | "v" ) wsp* vertical-lineto-argument-sequence
    bool vertical_lineto() {
        const void *save = savePosition();
        expectCmds('v', 'V');
        wsp_star();
        if (!vertical_lineto_argument_sequence()) {
            rewindToPosition(save);
            return false;
        }
        return true;
    }

    // HELPER: Pareses <coords>/2 worth of coordinate pairs.
    bool coord_pair_arguments(int coords) {
        const void *save = savePosition();
        double c[6];
        assert(coords > 0);
        assert(size_t(coords) <= countof(c));
        assert(coords % 2 == 0);
        if (!coordinate_pair(c[0], c[1])) {
            rewindToPosition(save);
            return false;
        }
        if (coords > 2) {
            comma_wsp_opt();
            if (!coordinate_pair(c[2], c[3])) {
                rewindToPosition(save);
                return false;
            }
            if (coords > 4) {
                comma_wsp_opt();
                if (!coordinate_pair(c[4], c[5])) {
                    rewindToPosition(save);
                    return false;
                }
            }
        }
        cmd.push_back(current_command);
        for (int i=0; i<coords; i++) {
            coord.push_back(float(c[i]));
        }
        return true;
    }

    // curveto-argument:
    //     coordinate-pair comma-wsp? coordinate-pair comma-wsp? coordinate-pair
    inline bool curveto_argument() {
        return coord_pair_arguments(6);
    }

    // curveto-argument-sequence:
    //     curveto-argument
    //     | curveto-argument comma-wsp? curveto-argument-sequence
    bool curveto_argument_sequence() {
        const void *save = savePosition();
        if (!curveto_argument()) {
            rewindToPosition(save);
            return false;
        }
        do {
            save = savePosition();
            comma_wsp_opt();
        } while(curveto_argument());
        rewindToPosition(save);
        return true;
    }

    // curveto:
    //     ( "C" | "c" ) wsp* curveto-argument-sequence
    bool curveto() {
        const void *save = savePosition();
        expectCmds('c', 'C');
        wsp_star();
        if (!curveto_argument_sequence()) {
            rewindToPosition(save);
            return false;
        }
        return true;
    }

    // smooth-curveto-argument:
    //     coordinate-pair comma-wsp? coordinate-pair
    inline bool smooth_curveto_argument() {
        return coord_pair_arguments(4);
    }

    // smooth-curveto-argument-sequence:
    //     smooth-curveto-argument
    //     | smooth-curveto-argument comma-wsp? smooth-curveto-argument-sequence
    bool smooth_curveto_argument_sequence() {
        const void *save = savePosition();
        if (!smooth_curveto_argument()) {
            rewindToPosition(save);
            return false;
        }
        do {
            save = savePosition();
            comma_wsp_opt();
        } while(smooth_curveto_argument());
        rewindToPosition(save);
        return true;
    }

    // smooth-curveto:
    //     ( "S" | "s" ) wsp* smooth-curveto-argument-sequence
    bool smooth_curveto() {
        const void *save = savePosition();
        expectCmds('s', 'S');
        wsp_star();
        if (!smooth_curveto_argument_sequence()) {
            rewindToPosition(save);
            return false;
        }
        return true;
    }

    // quadratic-bezier-curveto-argument:
    //     coordinate-pair comma-wsp? coordinate-pair
    inline bool quadratic_bezier_curveto_argument() {
        return coord_pair_arguments(4);
    }

    // quadratic-bezier-curveto-argument-sequence:
    //     quadratic-bezier-curveto-argument
    //     | quadratic-bezier-curveto-argument comma-wsp? 
    //         quadratic-bezier-curveto-argument-sequence
    bool quadratic_bezier_curveto_argument_sequence() {
        const void *save = savePosition();
        if (!quadratic_bezier_curveto_argument()) {
            rewindToPosition(save);
            return false;
        }
        do {
            save = savePosition();
            comma_wsp_opt();
        } while(quadratic_bezier_curveto_argument());
        rewindToPosition(save);
        return true;
    }

    // quadratic-bezier-curveto:
    //     ( "Q" | "q" ) wsp* quadratic-bezier-curveto-argument-sequence
    bool quadratic_bezier_curveto() {
        const void *save = savePosition();
        expectCmds('q', 'Q');
        wsp_star();
        if (!quadratic_bezier_curveto_argument_sequence()) {
            rewindToPosition(save);
            return false;
        }
        return true;
    }

    // smooth-quadratic-bezier-curveto-argument-sequence:
    //     coordinate-pair
    //     | coordinate-pair comma-wsp? smooth-quadratic-bezier-curveto-argument-sequence
    bool smooth_quadratic_bezier_curveto_argument_sequence() {
        const void *save = savePosition();
        if (!coord_pair_arguments(2)) {
            rewindToPosition(save);
            return false;
        }
        do {
            save = savePosition();
            comma_wsp_opt();
        } while(coord_pair_arguments(2));
        rewindToPosition(save);
        return true;
    }

    // smooth-quadratic-bezier-curveto:
    //     ( "T" | "t" ) wsp* smooth-quadratic-bezier-curveto-argument-sequence
    bool smooth_quadratic_bezier_curveto() {
        const void *save = savePosition();
        expectCmds('t', 'T');
        wsp_star();
        if (!smooth_quadratic_bezier_curveto_argument_sequence()) {
            rewindToPosition(save);
            return false;
        }
        return true;
    }

    // nonnegative-number:
    //     integer-constant
    //     | floating-point-constant
    bool nonnegative_number(double &value) {
        // NOTE: match floating-point-constant ahead of integer-constant
        const void *save = savePosition();
        if (floating_point_constant(value)) {
            return true;
        }
        rewindToPosition(save);
        if (integer_constant(value)) {
            return true;
        }
        rewindToPosition(save);
        return false;
    }


    // flag:
    //     "0" | "1"
    bool flag(float &flag_value) {
        switch (curChar()) {
        case '0':
        case '1':
            flag_value = curChar() - '0';
            advanceChar();
            return true;
        default:
            return false;
        }
    }

    // elliptical-arc-argument:
    //     nonnegative-number comma-wsp? nonnegative-number comma-wsp? 
    //         number comma-wsp flag comma-wsp? flag comma-wsp? coordinate-pair
    bool elliptical_arc_argument() {
        const void *save = savePosition();
        double rx, ry, x_axis_rotation, x, y;
        float large_arc_flag, sweep_flag;
        if (!nonnegative_number(rx)) {
            rewindToPosition(save);
            return false;
        }
        comma_wsp_opt();
        if (!nonnegative_number(ry)) {
            rewindToPosition(save);
            return false;
        }
        comma_wsp_opt();
        if (!number(x_axis_rotation)) {
            rewindToPosition(save);
            return false;
        }
        comma_wsp_opt();
        if (!flag(large_arc_flag)) {
            rewindToPosition(save);
            return false;
        }
        comma_wsp_opt();
        if (!flag(sweep_flag)) {
            rewindToPosition(save);
            return false;
        }
        comma_wsp_opt();
        if (!coordinate_pair(x, y)) {
            rewindToPosition(save);
            return false;
        }
        cmd.push_back(current_command);
        coord.push_back(float(rx));
        coord.push_back(float(ry));
        coord.push_back(float(x_axis_rotation));
        coord.push_back(large_arc_flag);
        coord.push_back(sweep_flag);
        coord.push_back(float(x));
        coord.push_back(float(y));
        return true;
    }

    // elliptical-arc-argument-sequence:
    //     elliptical-arc-argument
    //     | elliptical-arc-argument comma-wsp? elliptical-arc-argument-sequence
    bool elliptical_arc_argument_sequence() {
        const void *save = savePosition();
        if (!elliptical_arc_argument()) {
            rewindToPosition(save);
            return false;
        }
        do {
            save = savePosition();
            comma_wsp_opt();
        } while(elliptical_arc_argument());
        rewindToPosition(save);
        return true;
    }

    // elliptical-arc:
    //     ( "A" | "a" ) wsp* elliptical-arc-argument-sequence
    bool elliptical_arc() {
        const void *save = savePosition();
        expectCmds('a', 'A');
        wsp_star();
        if (!elliptical_arc_argument_sequence()) {
            rewindToPosition(save);
            return false;
        }
        return true;
    }

    // drawto-command:
    //     closepath
    //     | lineto
    //     | horizontal-lineto
    //     | vertical-lineto
    //     | curveto
    //     | smooth-curveto
    //     | quadratic-bezier-curveto
    //     | smooth-quadratic-bezier-curveto
    //     | elliptical-arc
    bool drawto_command() {
        const void *save = savePosition();
        switch (curChar()) {
        case 'z':
        case 'Z':
            if (closepath()) {
                return true;
            }
            break;
        case 'l':
        case 'L':
            if (lineto()) {
                return true;
            }
            break;
        case 'h':
        case 'H':
            if (horizontal_lineto()) {
                return true;
            }
            break;
        case 'v':
        case 'V':
            if (vertical_lineto()) {
                return true;
            }
            break;
        case 'c':
        case 'C':
            if (curveto()) {
                return true;
            }
            break;
        case 's':
        case 'S':
            if (smooth_curveto()) {
                return true;
            }
            break;
        case 'q':
        case 'Q':
            if (quadratic_bezier_curveto()) {
                return true;
            }
            break;
        case 't':
        case 'T':
            if (smooth_quadratic_bezier_curveto()) {
                return true;
            }
            break;
        case 'a':
        case 'A':
            if (elliptical_arc()) {
                return true;
            }
            break;
        default:  // unrecognized command
            break;
        }
        rewindToPosition(save);
        return false;
    }

    // drawto-commands:
    //     drawto-command
    //     | drawto-command wsp* drawto-commands
    bool drawto_commands() {
        const void *save = savePosition();
        if (!drawto_command()) {
            rewindToPosition(save);
            return false;
        }
        do {
            save = savePosition();
            wsp_star();
        } while(drawto_command());
        rewindToPosition(save);
        return true;
    }

    // moveto-drawto-command-group:
    //     moveto wsp* drawto-commands?
    bool moveto_drawto_command_group() {
        const void *save = savePosition();
        if (!moveto()) {
            rewindToPosition(save);
            return false;
        }
        wsp_star();
        save = savePosition();
        if (!drawto_commands()) {
            rewindToPosition(save);
        }
        return true;
    }

    // moveto-drawto-command-groups:
    //     moveto-drawto-command-group
    //     | moveto-drawto-command-group wsp* moveto-drawto-command-groups
    bool moveto_drawto_command_groups() {
        const void *save = savePosition();
        if (!moveto_drawto_command_group()) {
            rewindToPosition(save);
            return false;
        }
        do {
            save = savePosition();
            wsp_star();
        } while(moveto_drawto_command_group());
        rewindToPosition(save);
        return true;
    }

    //svg-path:
    //    wsp* moveto-drawto-command-groups? wsp*
    bool svg_path() {
        wsp_star();
        const void *save = savePosition();
        if (!moveto_drawto_command_groups()) {
            rewindToPosition(save);
        }
        wsp_star();
        // Expect end of string.
        if (!atEnd()) {
            return false;
        }
        return true;
    };

    // Construtor is private, use SVGPathParser::parse static method to parse.
    SVGPathParser(const char *input_, vector<char> &cmd_, vector<float> &coord_)
        : LimitedStringAccess<char>(input_, input_+strlen(input_))
        , cmd(cmd_)
        , coord(coord_)
        , current_command(0)
    {
        // Reset the output arrays
        cmd.clear();
        coord.clear();
    }

public:

    // Static member function provides one-step "front door" to the parser.
    static bool parse(const char *input, vector<char> &cmd, vector<float> &coord, int &error_position) {
        SVGPathParser parser(input, cmd, coord);
        bool success = parser.svg_path();
        error_position = parser.errorPosition();
        return success;
    }
};


int svg_path_parser(const char *input, vector<char> &c, vector<float> &v)
{
    int error_position;
    bool ok = SVGPathParser::parse(input, c, v, error_position);
    if (error_position >= 0) {
        printf("error_location = %d\n", error_position);
        printf("error begins: %s\n", &input[error_position]);
    }
    return ok;
}

bool too_different(float a, float b)
{
#if 1 // approximate (relative) match
    double ratio = a/b;

    if (ratio < 0.999999 || ratio > 1.000001) {
        return true;
    } else {
        return false;
    }
#else // exact match
    return a != b;
#endif
}

int parse_svg_path(const char *input, vector<char> &c, vector<float> &v)
{
#if 0
    return spirit_svg_path_parser(input, c, v);
#else

    int status = svg_path_parser(input, c, v);

#if 0
    // compare with boost spirit's parse...
    vector<char> spirit_c;
    vector<float> spirit_v;
    bool mismatch = false;

    int boost_status = spirit_svg_path_parser(input, spirit_c, spirit_v);

    if (spirit_c.size() != c.size()) {
        printf("command list mismatch: got %d, spirit got %d\n", c.size(), spirit_c.size());
        mismatch = true;
    }
    size_t cmd_size = min(spirit_c.size(), c.size());
    if (cmd_size > 0) {
        for (size_t i=0; i<cmd_size; i++) {
            if (c[i] != spirit_c[i]) {
                printf("cmd %d: mismatch, got %d, spririt got %d\n", i, c[i], spirit_c[i]);
                mismatch = true;
            }
        }
    }
    if (spirit_v.size() != v.size()) {
        printf("coord list mismatch: got %d, spirit got %d\n", v.size(), spirit_v.size());
        mismatch = true;
    }
    size_t coord_size = min(spirit_v.size(), v.size());
    if (coord_size > 0) {
        for (size_t i=0; i<coord_size; i++) {
            if (too_different(v[i], spirit_v[i])) {
                printf("coord %d: mismatch, got %f, spririt got %f\n", i, v[i], spirit_v[i]);
                mismatch = true;
            }
        }
    }

    if (mismatch) {
        c = spirit_c;
        v = spirit_v;
        return boost_status;
    }
#endif

    return status;
#endif
}

